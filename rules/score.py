"""
计分器 - 温州麻将番数计算

计分公式：
对方支付总额 = 胡牌倍数 × 连庄倍数 × 底分 × (1 + 财神个数)

财神分：
每有1个财神，额外获得 1 × 胡牌倍数 × 连庄倍数 × 底分

杠分：
杠一次得1分

连庄倍数：
第1次连庄 ×1
第2次连庄 ×2
第3次连庄 ×3
第4次连庄 ×4
第4次以后不再翻，换对家做庄
"""

from typing import List, Dict
from .caishen import CaishenEngine
from .win import WinChecker


class ScoreCalculator:
    """温州麻将计分器"""

    def __init__(self, caishen_id: int, base_score: int = 2):
        self.caishen = CaishenEngine(caishen_id)
        self.base_score = base_score  # 底分
        self.lian_zhuang_count = 0  # 连庄次数

    def calc_score(self, hand: List[int],
                   is_self_draw: bool = False,
                   is_qiang_gang: bool = False,
                   is_gang_kai: bool = False,
                   is_tian_hu: bool = False,
                   is_di_hu: bool = False,
                   is_qing_yi_se: bool = False,
                   is_ban_qing: bool = False,
                   is_dan_diao: bool = False,
                   is_qiu_cai_sheng: bool = False,
                   is_peng_peng_hu: bool = False,
                   is_double_caishen_gui_wei: bool = False) -> Dict:
        """
        计算得分

        Returns:
            {
                'base_score': int,       # 基础分
                'fan': int,              # 番数倍数
                'caishen_count': int,    # 财神数量
                'lian_zhuang': int,      # 连庄倍数
                'total_score': int,      # 总分
                'details': []            # 详情
            }
        """
        win_checker = WinChecker(self.caishen.caishen_id)
        win_result = win_checker.check_win(hand)

        if not win_result['can_win']:
            return {'base_score': 0, 'fan': 0, 'total_score': 0}

        caishen_count = self.caishen.count_caishen(hand)
        details = list(win_result.get('details', []))

        # ===== 1. 判断胡牌基础番数 =====
        fan = win_result.get('fan', 1)  # soft=1, hard=2, double_fan=4, special_8fan=8
        win_type = win_result.get('win_type', 'soft')

        # ===== 2. 检查双翻条件 (×4) =====
        is_double_fan = False

        # 天胡
        if is_tian_hu:
            fan = 4
            is_double_fan = True
            details.append('天胡')

        # 地胡
        if is_di_hu:
            fan = 4
            is_double_fan = True
            details.append('地胡')

        # 硬8对 (已经是fan=4)
        # 3财神+正常牌型 (已经是fan=4)

        # 碰碰胡
        if is_peng_peng_hu:
            if fan < 4:
                fan = 4
                is_double_fan = True
            details.append('碰碰胡')

        # 单吊
        if is_dan_diao:
            if not is_double_fan:  # 单吊不是双翻，是硬牌
                fan = max(fan, 2)
            details.append('单吊')

        # 全球神
        if is_qiu_cai_sheng:
            if not is_double_fan:
                fan = max(fan, 2)
            details.append('全球神')

        # 双财神归位
        if is_double_caishen_gui_wei:
            fan = 4
            is_double_fan = True
            details.append('双财神归位')

        # 半清
        if is_ban_qing:
            if not is_double_fan:
                fan = max(fan, 2)
            details.append('半清')

        # 清一色
        if is_qing_yi_se:
            fan = 4
            is_double_fan = True
            details.append('清一色')

        # 抢杠胡
        if is_qiang_gang:
            fan = 4
            is_double_fan = True
            details.append('抢杠')

        # 杠上开花
        if is_gang_kai:
            fan = 4
            is_double_fan = True
            details.append('杠上开花')

        # 自摸 (在双翻条件下保持×4，否则变成×2硬牌)
        if is_self_draw:
            if not is_double_fan:
                fan = max(fan, 2)
            details.append('自摸')

        # ===== 3. 连庄倍数 =====
        lian_zhuang = min(self.lian_zhuang_count + 1, 4)
        if self.lian_zhuang_count > 0:
            details.append(f'连庄{self.lian_zhuang_count + 1}次')

        # ===== 4. 计算总分 =====
        # 基础分 = 胡牌倍数 × 连庄倍数 × 底分
        base = fan * lian_zhuang * self.base_score

        # 财神加分 = 财神个数 × 胡牌倍数 × 连庄倍数 × 底分
        caishen_bonus = caishen_count * fan * lian_zhuang * self.base_score

        # 总分 = (基础分 + 财神加分)
        total_score = base + caishen_bonus

        return {
            'base_score': base,
            'fan': fan,
            'caishen_count': caishen_count,
            'lian_zhuang': lian_zhuang,
            'caishen_bonus': caishen_bonus,
            'total_score': total_score,
            'details': details,
            'formula': f'{fan} × {lian_zhuang} × {self.base_score} × (1 + {caishen_count}) = {total_score}'
        }

    def calc_gang_score(self, gang_type: str = 'ming_gang') -> int:
        """计算杠分 - 杠一次得1分"""
        return 1

    def on_win(self, is_self_draw: bool = False, is_liuju: bool = False) -> str:
        """
        胡牌后处理连庄
        闲家胡牌后下家坐庄，连庄次数归零
        流局时庄家连庄
        """
        if is_liuju:
            # 流局连庄
            self.lian_zhuang_count += 1
            if self.lian_zhuang_count >= 4:
                self.lian_zhuang_count = 0
                return 'change_zhuang'
        elif not is_self_draw:
            # 闲家胡牌，连庄归零，下家坐庄
            self.lian_zhuang_count = 0
        else:
            # 自摸也归零连庄
            self.lian_zhuang_count = 0
        return 'continue'

    def on_liuju(self):
        """流局后处理连庄"""
        self.lian_zhuang_count += 1
        if self.lian_zhuang_count >= 4:
            self.lian_zhuang_count = 0
            return 'change_zhuang'
        return 'continue'

    def reset_lian_zhuang(self):
        """重置连庄计数"""
        self.lian_zhuang_count = 0


# ===== 快捷函数 =====

def quick_calc_score(hand: List[int], caishen_id: int,
                     is_self_draw: bool = False,
                     lian_zhuang: int = 0,
                     base_score: int = 2,
                     **special_flags) -> Dict:
    """快速计算分数"""
    calc = ScoreCalculator(caishen_id, base_score)
    calc.lian_zhuang_count = lian_zhuang
    return calc.calc_score(hand, is_self_draw=is_self_draw, **special_flags)