import unittest
import ddl_build_module


class MyTestCase(unittest.TestCase):
    def test_create_insert_script(self):
        script = "INSERT INTO card.encoded (time,amount,vendor) VALUES(%s,%s,%s)"
        builder = ddl_build_module.DdlCommandBuilder("card")
        fields = ["time", "amount", "vendor"]
        built_script = builder.create_insert_into_encoded_or_feature_engineered_script(fields, "encoded")
        self.assertEqual(built_script, script)  # add assertion here

    def test_create_encoded_table_script(self):
        script="CREATE TABLE IF NOT EXISTS card.encoded(id BIGINT NOT NULL AUTO_INCREMENT,time DOUBLE PRECISION,amount DOUBLE PRECISION,vendor DOUBLE PRECISION,PRIMARY KEY (id)) ENGINE = InnoDB"
        builder = ddl_build_module.DdlCommandBuilder("card")
        fields = ["time", "amount", "vendor"]
        built_script = builder.build_create_encoded_or_engineered_table_script(fields, "encoded")
        self.assertEqual(script,built_script)


if __name__ == '__main__':
    unittest.main()
