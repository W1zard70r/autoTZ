from enum import Enum

class DataEnum(str, Enum):
    CHAT = "chat"
    DOCUMENT = "document"
    PLAIN_TEXT = "plain_text"
    GRAPHML = "graphml"

class TZSectionEnum(str, Enum):
    GENERAL = "general_info"
    STACK = "tech_stack"
    FUNCTIONAL = "functional_req"
    INTERFACE = "ui_ux"
    UNKNOWN = "uncategorized"

class NodeLabel(str, Enum):
    PERSON = "Person"
    COMPONENT = "Component"
    TASK = "Task"
    REQUIREMENT = "Requirement"
    CONCEPT = "Concept"

class EdgeRelation(str, Enum):
    ASSIGNED_TO = "ASSIGNED_TO"
    DEPENDS_ON = "DEPENDS_ON"
    RELATES_TO = "RELATES_TO"
    AGREES_WITH = "AGREES_WITH"
    MENTIONS = "MENTIONS"