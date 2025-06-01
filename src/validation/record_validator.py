"""
牌譜検証モジュール
牌譜の構造と内容の詳細検証を提供
"""

import re
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from ..utils.tile_definitions import TileDefinitions
from ..utils.logger import get_logger


class RecordFormat(Enum):
    """牌譜形式"""
    TENHOU = "tenhou"
    UNKNOWN = "unknown"


@dataclass
class ValidationError:
    """検証エラー"""
    error_type: str
    message: str
    location: Optional[str] = None
    severity: str = "error"  # error, warning, info


class RecordValidator:
    """牌譜検証クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = get_logger(__name__)
        self.tile_definitions = TileDefinitions()
        
        # 検証ルール
        self.validation_rules = self._setup_validation_rules()
        
        self.logger.info("RecordValidator initialized")
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """検証ルールを設定"""
        return {
            'tenhou_rules': {
                'required_elements': ['mjloggm', 'INIT'],
                'valid_tags': ['INIT', 'T', 'D', 'U', 'N', 'REACH', 'AGARI', 'RYUUKYOKU'],
                'seed_pattern': r'\d+,\d+,\d+,\d+,\d+,\d+',
                'ten_pattern': r'\d+,\d+,\d+,\d+',
                'hai_pattern': r'\d+(,\d+)*'
            },
            'tile_rules': {
                'valid_tile_count': 136,  # 麻雀牌の総数
                'hand_size_limits': {
                    'min': 13,
                    'max': 14
                },
                'discard_limits': {
                    'min': 0,
                    'max': 25
                }
            }
        }
    
    def detect_format(self, record_data: Any) -> RecordFormat:
        """牌譜形式を検出（天鳳JSON専用）"""
        try:
            if isinstance(record_data, dict):
                # 天鳳JSON形式の場合
                if 'xml_content' in record_data:
                    xml_content = record_data['xml_content']
                    if '<mjloggm' in xml_content and 'INIT' in xml_content:
                        return RecordFormat.TENHOU
                # 直接天鳳JSON形式の場合
                elif any(key in record_data for key in ['title', 'name', 'rule', 'log']):
                    return RecordFormat.TENHOU
            
            elif isinstance(record_data, str):
                # XML文字列の場合
                if '<mjloggm' in record_data and 'INIT' in record_data:
                    return RecordFormat.TENHOU
            
            return RecordFormat.UNKNOWN
            
        except Exception as e:
            self.logger.error(f"Failed to detect record format: {e}")
            return RecordFormat.UNKNOWN
    
    
    def validate_tenhou_format(self, record_data: str) -> List[ValidationError]:
        """天鳳形式の検証"""
        errors = []
        rules = self.validation_rules['tenhou_rules']
        
        try:
            # 基本的なXML構造の確認
            if not record_data.strip().startswith('<'):
                errors.append(ValidationError(
                    error_type="invalid_format",
                    message="Not a valid XML format",
                    location="root"
                ))
                return errors
            
            # 必須要素の確認
            for element in rules['required_elements']:
                if element not in record_data:
                    errors.append(ValidationError(
                        error_type="missing_element",
                        message=f"Required element missing: {element}",
                        location="xml"
                    ))
            
            # INIT要素の検証
            init_matches = re.findall(r'<INIT[^>]*>', record_data)
            for i, init_match in enumerate(init_matches):
                init_errors = self._validate_tenhou_init(init_match, i, rules)
                errors.extend(init_errors)
            
            # アクション要素の検証
            action_patterns = [
                (r'<T\d+/>', 'draw'),
                (r'<D\d+/>', 'discard'),
                (r'<U\d+/>', 'draw_from_wall'),
                (r'<N[^>]*>', 'call'),
                (r'<REACH[^>]*>', 'riichi'),
                (r'<AGARI[^>]*>', 'win'),
                (r'<RYUUKYOKU[^>]*>', 'draw_game')
            ]
            
            for pattern, action_type in action_patterns:
                matches = re.findall(pattern, record_data)
                for match in matches:
                    # 基本的な形式チェック
                    if not match.endswith('/>') and not match.endswith('>'):
                        errors.append(ValidationError(
                            error_type="invalid_xml",
                            message=f"Invalid XML tag format: {match}",
                            location=f"xml.{action_type}"
                        ))
        
        except Exception as e:
            errors.append(ValidationError(
                error_type="validation_error",
                message=f"Tenhou validation error: {str(e)}",
                location="xml"
            ))
        
        return errors
    
    def _validate_tenhou_init(self, init_element: str, 
                             init_index: int, rules: Dict[str, Any]) -> List[ValidationError]:
        """天鳳INIT要素の検証"""
        errors = []
        location_prefix = f"xml.INIT[{init_index}]"
        
        try:
            # seed属性の確認
            seed_match = re.search(r'seed="([^"]*)"', init_element)
            if seed_match:
                seed_value = seed_match.group(1)
                if not re.match(rules['seed_pattern'], seed_value):
                    errors.append(ValidationError(
                        error_type="invalid_seed",
                        message=f"Invalid seed format: {seed_value}",
                        location=f"{location_prefix}.seed"
                    ))
            
            # ten属性の確認
            ten_match = re.search(r'ten="([^"]*)"', init_element)
            if ten_match:
                ten_value = ten_match.group(1)
                if not re.match(rules['ten_pattern'], ten_value):
                    errors.append(ValidationError(
                        error_type="invalid_ten",
                        message=f"Invalid ten format: {ten_value}",
                        location=f"{location_prefix}.ten"
                    ))
            
            # hai属性の確認（手牌）
            for i in range(4):  # 4人分
                hai_match = re.search(f'hai{i}="([^"]*)"', init_element)
                if hai_match:
                    hai_value = hai_match.group(1)
                    if not re.match(rules['hai_pattern'], hai_value):
                        errors.append(ValidationError(
                            error_type="invalid_hai",
                            message=f"Invalid hai{i} format: {hai_value}",
                            location=f"{location_prefix}.hai{i}"
                        ))
        
        except Exception as e:
            errors.append(ValidationError(
                error_type="validation_error",
                message=f"INIT validation error: {str(e)}",
                location=location_prefix
            ))
        
        return errors
    
    def validate_tile_consistency(self, record_data: Any) -> List[ValidationError]:
        """牌の一貫性検証"""
        errors = []
        
        try:
            # 牌の使用状況を追跡
            tile_usage = {}
            
            # 天鳳形式から牌を抽出
            format_type = self.detect_format(record_data)
            
            if format_type == RecordFormat.TENHOU:
                tiles = self._extract_tiles_from_tenhou(record_data)
            else:
                return [ValidationError(
                    error_type="unknown_format",
                    message="天鳳形式以外はサポートされていません",
                    location="root"
                )]
            
            # 牌の使用回数をカウント
            for tile in tiles:
                tile_usage[tile] = tile_usage.get(tile, 0) + 1
            
            # 牌の使用回数をチェック
            for tile, count in tile_usage.items():
                max_count = self.tile_definitions.get_max_tile_count(tile)
                if count > max_count:
                    errors.append(ValidationError(
                        error_type="tile_overuse",
                        message=f"Tile {tile} used {count} times (max: {max_count})",
                        location="tiles"
                    ))
        
        except Exception as e:
            errors.append(ValidationError(
                error_type="validation_error",
                message=f"Tile consistency validation error: {str(e)}",
                location="tiles"
            ))
        
        return errors
    
    
    def _extract_tiles_from_tenhou(self, record_data: str) -> List[str]:
        """天鳳形式から牌を抽出"""
        tiles = []
        
        # 簡易的な牌抽出（実際の実装ではより詳細な解析が必要）
        # 数値から牌名への変換が必要
        tile_patterns = [
            r'<T(\d+)/>',
            r'<D(\d+)/>',
            r'<U(\d+)/>'
        ]
        
        for pattern in tile_patterns:
            matches = re.findall(pattern, record_data)
            for match in matches:
                # 数値を牌名に変換（簡易版）
                tile_name = self._convert_tenhou_tile_id(int(match))
                if tile_name:
                    tiles.append(tile_name)
        
        return tiles
    
    def _convert_tenhou_tile_id(self, tile_id: int) -> Optional[str]:
        """天鳳の牌IDを牌名に変換"""
        try:
            # 天鳳の牌ID変換ルール（簡易版）
            if 0 <= tile_id <= 35:
                suit = tile_id // 9
                number = (tile_id % 9) + 1
                
                if suit == 0:
                    return f"{number}m"
                elif suit == 1:
                    return f"{number}p"
                elif suit == 2:
                    return f"{number}s"
                elif suit == 3:
                    honors = ["東", "南", "西", "北", "白", "發", "中"]
                    if number <= len(honors):
                        return honors[number - 1]
            
            return None
            
        except Exception:
            return None