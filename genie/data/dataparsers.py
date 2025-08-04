from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig, Blender

@dataclass
class GENIEBlenderDataParserConfig(BlenderDataParserConfig):
    """Configuration for GENIE Blender data parser."""

    _target: Type = field(default_factory=lambda: GENIEBlender)

class GENIEBlender(Blender):
    """GENIE Blender data parser.

    This class extends the BlenderDataParser to handle GENIE-specific data parsing.
    """

    def __init__(self, config: GENIEBlenderDataParserConfig):

        config.ply_path ="sparse_pc.ply"
        super().__init__(config)
