from unittest import TestCase

import pytest

from acro.restricted import RestrictedDataFrame

class TestEmpty:
    rdf_empty = RestrictedDataFrame()

    def test_str(self):
        with pytest.raises(AttributeError) as e_info:
            str(self.rdf_empty)

    def test_repr(self):
        with pytest.raises(AttributeError) as e_info:
            repr(self.rdf_empty)

    def test_head(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.head()

    def test_tail(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.tail()

    # serializtion `to_` methods
    def test_to_orc(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_orc()

    def test_to_parquet(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_parquet()

    def test_to_pickle(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_pickle("empty_rdf.pck")

    def test_to_hdf(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_hdf("empty_rdf.h5", key="rdf")

    def test_to_sql(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_sql("empty_rdf", None)

    def test_to_csv(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_csv()

    def test_to_excel(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_excel()

    def test_to_json(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_json()

    def test_to_html(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_html()

    def test_to_feather(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_feather("empty_rdf.fea")

    def test_to_latex(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_latex()

    def test_to_stata(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_stata("empty_rdf.sta")

    def test_to_clipboard(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_clipboard()

    def test_to_markdown(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_markdown()

    def test_to_records(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_records()

    def test_to_string(self):
        with pytest.raises(AttributeError) as e_info:
            self.rdf_empty.to_string()





