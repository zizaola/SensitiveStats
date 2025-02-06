# --- Example Code to Analyze ---
from acro import restricted
from acro.restricted import (
    RestrictedDataFrame,
)

# Creating instances of RestrictedDataFrame in two ways:
rdf1 = RestrictedDataFrame({"a": [1, 2, 3]})
rdf2 = RestrictedDataFrame({"b": [4, 5, 6]})
rdf3 = restricted.RestrictedDataFrame({"c": [7, 8, 9]})

not_a_df = [10, 20, 30]

# Using .at on rdf1; this should be detected.
value = rdf1.at[0, "a"]
value3 = rdf3.at[1, "c"]

value4 = rdf1.loc[0]
