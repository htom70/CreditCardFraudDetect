import server
import unittest

encoding_parametrs = {
    "card_number": "label_encoder",
    "transaction_type": "omit",
}

detailed_information_about_table = {
    "fields": [
        {
            "name": "id",
            "other_eligible_type_for_encoding": [],
            "type": "bigint"
        },
        {
            "name": "card_number",
            "other_eligible_type_for_encoding": [
                "float"
            ],
            "type": "varchar"
        },
        {
            "name": "transaction_type",
            "other_eligible_type_for_encoding": [],
            "type": "int"
        },
        {
            "name": "timestamp",
            "other_eligible_type_for_encoding": [],
            "type": "datetime"
        },
        {
            "name": "amount",
            "other_eligible_type_for_encoding": [],
            "type": "int"
        },
        {
            "name": "currency_name",
            "other_eligible_type_for_encoding": [],
            "type": "varchar"
        },
        {
            "name": "response_code",
            "other_eligible_type_for_encoding": [],
            "type": "int"
        },
        {
            "name": "country_name",
            "other_eligible_type_for_encoding": [],
            "type": "varchar"
        },
        {
            "name": "vendor_code",
            "other_eligible_type_for_encoding": [
                "float",
                "int"
            ],
            "type": "varchar"
        },
        {
            "name": "fraud",
            "other_eligible_type_for_encoding": [],
            "type": "int"
        }
    ],
    "fraud_candidates": [
        {
            "fraud_number": 601,
            "name": "fraud",
            "no_fraud_number": 11314
        }
    ],
    "primary_key": [
        "id"
    ],
    "record_number": 11915
}


def get_fields_from_detailed_information(detailed_information_about_table):
    result = list()
    complex_fields = detailed_information_about_table.get("fields")
    for complex_field_item in complex_fields:
        field_name = complex_field_item.get("name")
        result.append(field_name)
    return result


class TestServer(unittest.TestCase):
    def test_get_number_of_int_and_float_typed_fields(self):
        number_of_numerical_fields = server.get_number_of_int_and_float_typed_fields(
            detailed_information_about_table.get("fields"))
        self.assertEqual(number_of_numerical_fields, 5)

    def test_to_many_fraud_type_in_parameters_then_throws_exception(self):
        encoding_parameters = {
            "fraud": "fraud_type",
            "transaction_type": "fraud_type"
        }
        with self.assertRaises(server.TooManyFraudTypeFieldException):
            fields = get_fields_from_detailed_information(detailed_information_about_table)
            server.check_number_of_fraud_type_field(fields, encoding_parameters)

    def test_if_no_fraud_type_in_parameters_then_then_throws_exception(self):
        encoding_parameters = {}
        fields = get_fields_from_detailed_information(detailed_information_about_table)
        with self.assertRaises(server.NoFraudTypeFieldException):
            server.check_number_of_fraud_type_field(fields, encoding_parameters)

    def test_if_use_unknown_encoding_type_then_check_encoding_parameters_throws_exception(self):
        encoding_parameters = {
            "vendor_code": "double"
        }
        with self.assertRaises(server.EncodingNotApplicableException):
            server.check_encoding_parameters(encoding_parameters, detailed_information_about_table)

    def test_if_use_wrong_encoding_type_then_check_encoding_parameters_throws_exception(self):
        encoding_parameters = {
            "vendor_code": "julian"
        }
        with self.assertRaises(server.EncodingNotApplicableException):
            server.check_encoding_parameters(encoding_parameters, detailed_information_about_table)

    def test_if_feature_engineering_not_complete_then_check_throws_exception(self):
        feature_engineering_parameters = {
            "feature_name_for_engineering": "id",
            "time_base_field_name": "timestamp"
        }
        encoding_parameters = {
            "card_number": "label_encoder",
            "transaction_type": "omit",
        }
        with self.assertRaises(server.FeatureEngineeringNotApplicableException):
            server.check_if_feature_engineering_applicable(detailed_information_about_table, encoding_parameters,
                                                           feature_engineering_parameters)

    def test_if_use_not_numerical_type_for_engineering_then_check_throws_exception(self):
        feature_engineering_parameters = {
            "feature_name_for_engineering": "id",
            "intervals": [3, 7],
            "time_base_field_name": "timestamp"
        }
        encoding_parameters = {
            "card_number": "label_encoder",
            "transaction_type": "omit",
        }

        with self.assertRaises(server.FeatureEngineeringNotApplicableException):
            server.check_if_feature_engineering_applicable(detailed_information_about_table, encoding_parameters,
                                                           feature_engineering_parameters)

    def test_if_use_not_datetime_type_field_for_engineering_time_base_then_check_throws_exception(self):
        feature_engineering_parameters = {
            "feature_name_for_engineering": "amount",
            "intervals": [3, 7],
            "time_base_field_name": "amount"
        }
        encoding_parameters = {
            "card_number": "label_encoder",
            "transaction_type": "omit",
        }

        with self.assertRaises(server.FeatureEngineeringNotApplicableException):
            server.check_if_feature_engineering_applicable(detailed_information_about_table, encoding_parametrs,
                                                           feature_engineering_parameters)

    def test_if_use_not_empty_interval_list_for_engineering_time_base_then_check_throws_exception(self):
        feature_engineering_parameters = {
            "feature_name_for_engineering": "amount",
            "intervals": [],
            "time_base_field_name": "timestamp"
        }
        encoding_parameters = {
            "card_number": "label_encoder",
            "transaction_type": "omit",
        }

        with self.assertRaises(server.FeatureEngineeringNotApplicableException):
            server.check_if_feature_engineering_applicable(detailed_information_about_table, encoding_parameters,
                                                           feature_engineering_parameters)

    def test_if_omit_feature_engineering_field_during_encoding_then_check_throws_exception(self):
        feature_engineering_parameters = {
            "feature_name_for_engineering": "amount",
            "intervals": [2, 5],
            "time_base_field_name": "timestamp"
        }
        encoding_parameters = {
            "amount": "omit",
            "transaction_type": "omit"
        }

        with self.assertRaises(server.FeatureEngineeringNotApplicableException):
            server.check_if_feature_engineering_applicable(detailed_information_about_table, encoding_parameters,
                                                           feature_engineering_parameters)

    def test_if_omit_feature_engineering_time_base_field_during_encoding_then_check_throws_exception(self):
        feature_engineering_parameters = {
            "feature_name_for_engineering": "amount",
            "intervals": [2, 5],
            "time_base_field_name": "timestamp"
        }
        encoding_parameters = {
            "timestamp": "omit",
            "transaction_type": "omit"
        }

        with self.assertRaises(server.FeatureEngineeringNotApplicableException):
            server.check_if_feature_engineering_applicable(detailed_information_about_table, encoding_parameters,
                                                           feature_engineering_parameters)

    def test_get_fraud_type_field_name(self):
        encoding_parameters = {
            "card_number": "float",
            "timestamp": "julian",
            "currency_name": "label_encoder",
            "fraud": "fraud_type",
            "country_name": "omit",
            "vendor_code": "label_encoder"
        }
        fraud_type_field_name=server.get_fraud_type_field_name(encoding_parameters)
        self.assertEqual(fraud_type_field_name, "fraud")


if __name__ == "__main__":
    unittest.main()
