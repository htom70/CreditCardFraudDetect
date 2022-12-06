import logging

COMMA = ","
SPACE = " "
NEW_LINE = "\n"
BEGIN_BRACKET = "("
END_BRACKET = ")"
DOT = "."


class DdlCommandBuilder:
    def __init__(self, schema_name):
        self.logger = logging.getLogger("train.server.database.ddl")
        self.schema_name = schema_name

    def build_create_encoded_or_engineered_table_script(self, field_names, table_name):
        script = "CREATE TABLE IF NOT EXISTS" + SPACE + self.schema_name + DOT + table_name
        script += BEGIN_BRACKET
        script += "id BIGINT NOT NULL AUTO_INCREMENT" + COMMA
        for field_name in field_names:
            script += field_name + SPACE + "DOUBLE PRECISION" + COMMA
        script += "PRIMARY KEY (id)"
        script += END_BRACKET + SPACE
        script += "ENGINE = InnoDB"
        return script

    def create_insert_into_encoded_or_feature_engineered_script(self, field_names, table_name):
        script = "INSERT INTO" + SPACE + self.schema_name + DOT + table_name + SPACE
        script += BEGIN_BRACKET
        field_number = len(field_names)
        for i in range(field_number):
            script += field_names[i]
            if i != field_number - 1:
                script += COMMA
        script += END_BRACKET + SPACE
        script += "VALUES"
        script += BEGIN_BRACKET
        for i in range(field_number):
            script += "%s"
            if i != field_number - 1:
                script += COMMA
        script += END_BRACKET
        return script

    def create_insert_into_encoded_or_feature_engineered_script_for_prediction(self, field_names, table_name,
                                                                               fraud_type_field_name):
        script = "INSERT INTO" + SPACE + self.schema_name + DOT + table_name + SPACE
        script += BEGIN_BRACKET
        field_names.remove(fraud_type_field_name)
        field_number = len(field_names)
        for i in range(field_number):
            script += field_names[i]
            if i != field_number - 1:
                script += COMMA
        script += END_BRACKET + SPACE
        script += "VALUES"
        script += BEGIN_BRACKET
        for i in range(field_number):
            script += "%s"
            if i != field_number - 1:
                script += COMMA
        script += END_BRACKET
        return script
