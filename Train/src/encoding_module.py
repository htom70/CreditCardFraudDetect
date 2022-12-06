import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d

PRIMARY_KEY = "primary_key"
OMIT = "omit"
FRAUD_TYPE = "fraud_type"
JULIAN = "julian"
LABEL_ENCODER = "label_encoder"
FLOAT = "float"
INT = "int"


class NotImplementedEncodingException(Exception):
    pass


class DataBaseEncoder:
    def __init__(self, dataset, original_fields):
        self.logger = logging.getLogger("train.server.database.encoding")
        self.original_dataset = dataset
        self.original_fields = original_fields

    def encode(self, encoding_parameters, detailed_information_about_table):
        primary_key_field_name = detailed_information_about_table.get(PRIMARY_KEY)
        encoder_by_field_name = dict()
        encoded_fields = self.build_encoded_field_names(encoding_parameters, primary_key_field_name)
        number_of_records = len(self.original_dataset)
        number_of_fields = len(encoded_fields)
        encoded_array = np.empty([number_of_records, number_of_fields])
        for field in encoded_fields:
            field_type_before_encoding = self.get_field_type_before_encoding(field, detailed_information_about_table)
            column_index_in_original_dataset = self.original_fields.index(field)
            column_index_in_encoded_array = encoded_fields.index(field)
            encoding_parameter = encoding_parameters.get(field)
            if encoding_parameter is None:
                if self.is_num_type(field_type_before_encoding):
                    encoded_array[:,
                    column_index_in_encoded_array: column_index_in_encoded_array + 1] = self.original_dataset[:,
                                                                                        column_index_in_original_dataset: column_index_in_original_dataset + 1]
                    continue
            elif encoding_parameter == FRAUD_TYPE:
                encoded_array[:,
                column_index_in_encoded_array: column_index_in_encoded_array + 1] = self.original_dataset[:,
                                                                                    column_index_in_original_dataset: column_index_in_original_dataset + 1]
                continue
            elif encoding_parameter == JULIAN:
                encoded_array[:,
                column_index_in_encoded_array: column_index_in_encoded_array + 1] = self.convert_timestamp_to_julian(column_index_in_original_dataset)
                continue
            elif encoding_parameter == LABEL_ENCODER:
                converted_column, encoder = self.convert_with_label_encoder(column_index_in_original_dataset)
                encoded_array[:, column_index_in_encoded_array: column_index_in_encoded_array + 1] = converted_column
                encoder_by_field_name[field] = encoder
                continue
            elif encoding_parameter == FLOAT:
                encoded_array[:,
                column_index_in_encoded_array: column_index_in_encoded_array + 1] = self.convert_string_to_float(column_index_in_original_dataset)
                continue
            elif encoding_parameter == INT:
                encoded_array[:,
                column_index_in_encoded_array: column_index_in_encoded_array + 1] = self.convert_string_to_int(column_index_in_original_dataset)
                continue
            else:
                raise NotImplementedEncodingException(
                    {"message": "The given encoding not implemented", "field_name": field,
                     "encoding type": encoding_parameter})
        return encoded_array, encoder_by_field_name, encoded_fields

    # 3.1.5.1
    def build_encoded_field_names(self, encoding_parameters, primary_key_field_name):
        feature_fields = list()
        fraud_type_field = None
        number_of_original_fields = len(self.original_fields)
        for index in range(number_of_original_fields):
            field = self.original_fields[index]
            if field == primary_key_field_name:
                continue  # id-re nincs szükség auto-increment miatt
            elif encoding_parameters.get(field) == FRAUD_TYPE:
                fraud_type_field = field
                continue
            elif encoding_parameters.get(field) == OMIT:
                continue  # elhagyandó
            feature_fields.append(field)
        if fraud_type_field is not None:
            feature_fields.append(fraud_type_field)
        return feature_fields

    def get_field_type_before_encoding(self, field, detailed_information_about_table):
        result = None
        for field_property in detailed_information_about_table.get("fields"):
            if field == field_property.get("name"):
                result = field_property.get("type")
        return result

    def is_num_type(self, item):
        integer_types = ("integer", "int", "smallint")
        float_types = ("float", "double")
        if item in integer_types or item in float_types:
            return True
        else:
            return False

    def convert_timestamp_to_julian(self, original_index):
        converted_time_stamp_datas = list()
        for timestamp in self.original_dataset[:, original_index:original_index + 1]:
            t = timestamp[0]
            ts = pd.Timestamp(t)
            converted_time_stamp_to_julian = ts.to_julian_date()
            converted_time_stamp_datas.append(converted_time_stamp_to_julian)
        converted_time_stamp_data_array = np.array(converted_time_stamp_datas)
        reshaped_array = converted_time_stamp_data_array.reshape(-1, 1)
        return reshaped_array

    def convert_with_label_encoder(self, original_index):
        strings = self.original_dataset[:, original_index:original_index + 1]
        encoder = LabelEncoder()
        modified_strings = column_or_1d(strings)
        encoder.fit(modified_strings)
        transformed_array = encoder.transform(modified_strings)
        reshaped_transformed_array = transformed_array.reshape(-1, 1)
        return reshaped_transformed_array, encoder

    def convert_string_to_float(self, original_index):
        strings = self.original_dataset[:, original_index:original_index + 1]
        floats = np.array(strings, dtype=float)
        reshaped_array = floats.reshape(-1, 1)
        return reshaped_array

    def convert_string_to_int(self, original_index):
        strings = self.original_dataset[:, original_index:original_index + 1]
        ints = np.array(strings, dtype=int)
        reshaped_array = ints.reshape(-1, 1)
        return reshaped_array

    #  új metódusok vége

    def convertCardNumberStringToFloat(self, array):
        cardNumberStrings = array[:, 1:2]
        # cardEncoder=LabelEncoder()
        # modifiedCardNumbers=column_or_1d(cardNumberStrings)
        # cardEncoder.fit(modifiedCardNumbers)
        # encodedCardNumbers=cardEncoder.transform(modifiedCardNumbers)
        # reshapedCardNumbers=encodedCardNumbers.reshape(-1,1)
        cardNumberFloats = np.array(cardNumberStrings, dtype=float)
        reshapedCardNumberIntegers = cardNumberFloats.reshape(-1, 1)
        array[:, 1:2] = reshapedCardNumberIntegers
        # array[:,1:2]=reshapedCardNumbers

    def convertVendorCodeStringToFloat(self, array):
        vendorCodeStrings = array[:, 8:9]
        vendorCodeIntegers_as_int_type = vendorCodeStrings.astype(np.int)
        vendorCodeIntegers_as_float_type = vendorCodeStrings.astype(np.float)
        vendorCodeIntegers = np.array(vendorCodeStrings, dtype=int)
        reshapedVendorCodeIntegers = vendorCodeIntegers.reshape(-1, 1)
        array[:, 8:9] = reshapedVendorCodeIntegers

    def convertCountryFeature(self, array):
        countries = array[:, 7:8]
        countryEncoder = LabelEncoder()
        modifiedCountries = column_or_1d(countries)
        countryEncoder.fit(modifiedCountries)
        encodedCountries = countryEncoder.transform(modifiedCountries)
        reshapedEncodedCountries = encodedCountries.reshape(-1, 1)
        array[:, 7:8] = reshapedEncodedCountries
        return countryEncoder

    def convertCurrencyFeature(self, array):
        currenciesArray = array[:, 5:6]
        currencyEncoder = LabelEncoder()
        currencyEncoder.fit(currenciesArray)
        encodedCurrencies = currencyEncoder.transform(currenciesArray)
        reshapedEncodedCurrencies = encodedCurrencies.reshape(-1, 1)
        array[:, 5:6] = reshapedEncodedCurrencies
        return currencyEncoder

    def saveData(self, dataSet, connection, dataBaseName):
        valuesArray = dataSet[:, 1:]
        cursor = connection.cursor()
        sqlUseQuery = "USE " + dataBaseName
        cursor.execute(sqlUseQuery)
        sqlInsertQuery = "INSERt INTO encoded_transaction (card_number,transaction_type,timestamp,amount,currency_name,response_code,country_name,vendor_code,fraud) VALUES " \
                         "(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        length = len(valuesArray)
        bound = 1000
        if length > bound:
            numberOfPartArray = int(length / bound)
            numberOfRestDatas = length - numberOfPartArray * bound
            for i in range(0, numberOfPartArray, 1):
                tempArray = valuesArray[i * bound:(i + 1) * bound, :]
                valueList = list()
                for record in tempArray:
                    valueList.append(tuple(record))
                cursor.executemany(sqlInsertQuery, valueList)
                connection.commit()
            tempArray = valuesArray[(numberOfPartArray) * bound:(numberOfPartArray) * bound + numberOfRestDatas, :]
            valueList = list()
            for record in tempArray:
                valueList.append(tuple(record))
            cursor.executemany(sqlInsertQuery, valueList)
            connection.commit()
        else:
            valueList = list()
            for record in valuesArray:
                valueList.append(tuple(record))
            cursor.executemany(sqlInsertQuery, valueList)
            connection.commit()
        cursor.close()

    def saveEncoder(self, connection, databaseName):
        cursor = connection.cursor()
        sqlUseQuery = "USE " + databaseName
        cursor.execute(sqlUseQuery)
        sqlInsertQuery = "INSERT INTO encoder (encoder_name,encoder_object) VALUES (%s,%s)"
        pickledCurrencyEncoder = pickle.dumps(self.currencyEncoder)
        cursor.execute(sqlInsertQuery, ("currency_encoder", pickledCurrencyEncoder))
        pickledCountryEncoder = pickle.dumps(self.countryEncoder)
        cursor.execute(sqlInsertQuery, ("country_encoder", pickledCountryEncoder))
        connection.commit()
        cursor.close()
