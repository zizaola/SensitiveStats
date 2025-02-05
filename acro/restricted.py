import pandas as pd
import inspect

class RestrictedDataFrame(pd.DataFrame):
    # @classmethod
    # def raise_attrerror(cls):
    #     raise ArithmeticError

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        method2remove = ["__str__", "__repr_html", "tail", "head"]
        for method_name in method2remove:
            setattr(RestrictedDataFrame, method_name, self.raise_attrerror)

        serialization_methods = ["to_orc", "to_pickle", "to_parquet", "to_hdf",
                                 "to_sql", "to_csv", "to_excel", "to_json", "to_html",
                                 "to_feather", "to_latex", "to_stata", "to_clipboard,",
                                 "to_markdown", "to_records", "to_string"]
        for method_name in serialization_methods:
            setattr(RestrictedDataFrame, method_name, self.raise_attrerror)

        # attrs = ["style"]
        attrs = []
        for attr in attrs:
            delattr(self, attr)

    def raise_attrerror(self, *args, **kwargs):
        """Overwrite display method.

        Raises
        ------
            AttributeError: _description_
        """
        raise AttributeError("Restricted DataFrame can't show data.")



    # def __getattribute__(self, name):
    #     print(name)
    #     method2remove = ['__str__', '__repr_html', 'tail', "head"]
    #     if name in method2remove:
    #         print("enter name")
    #         raise ArithmeticError

    #     # Default behaviour
    #     return self.__getattr__(name)
        
        # def __new__(cls, *args, **kwargs):
        # return super().__new__(cls, *args, **kwargs)
    
    # def _repr_(self):
    #     '''overwrite display method'''
    #     return 'Operation overloaded for __repr__ in restricted version'

    # def _repr_html_(self):
    #     '''overwrite display method'''
    #     return 'Operation overloaded for _repr_html_ in restricted version'

        
    # def __str__(self):
    #     '''overwrite display method'''
    #     return 'Operation overloaded for __str__ in restricted version'

 