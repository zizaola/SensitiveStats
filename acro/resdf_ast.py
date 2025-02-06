"""Analise python code to forbid use of some methods in RestrictedDataFrame."""

import ast


class RestrictedDataFrameAnalyzer(ast.NodeVisitor):
    MODULE_NAME = "restricted"
    CLASS_NAME = "RestrictedDataFrame"
    FORBIDDEN_METHODS = ["at", "loc"]

    def __init__(self):
        # Set to record names of variables assigned as RestrictedDataFrame instances.
        self.restricted_dfs = set()
        # Dictionary to record RestrictedDataFrame variables and the line numbers where .at is used.
        self.forbidden_usage = {}

    def visit_Assign(self, node):
        """Visit assignment nodes to find RestrictedDataFrames.

        Look for assignments like:
            rdf = restricted.RestrictedDataFrame(...)
        or
            rdf = RestrictedDataFrame(...)
        """
        if isinstance(node.value, ast.Call):
            func = node.value.func
            # Check if the call is of the form restricted.RestrictedDataFrame(...)
            if isinstance(func, ast.Attribute) and func.attr == self.__class__.CLASS_NAME:
                if isinstance(func.value, ast.Name) and func.value.id == self.__class__.MODULE_NAME:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.restricted_dfs.add(target.id)
            # Alternatively, if RestrictedDataFrame is used directly (e.g., after a direct import)
            elif isinstance(func, ast.Name) and func.id == self.__class__.CLASS_NAME:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.restricted_dfs.add(target.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Visit attribute nodes to find forbidden methods.

        Look for attribute accesses, and if the attribute is 'at'
        and the object is a known RestrictedDataFrame variable, record the usage.
        """
        if node.attr in self.__class__.FORBIDDEN_METHODS:
            # Check if the object on which a forbidden method is accessed is a variable name.
            if isinstance(node.value, ast.Name) and node.value.id in self.restricted_dfs:
                self.forbidden_usage.setdefault(node.value.id, []).append([node.lineno, node.attr])
        self.generic_visit(node)


code = open("./code_to_analyze.py").read()

# Parse the code into an AST.
tree = ast.parse(code)

# Create an analyzer and run it over the AST.
analyzer = RestrictedDataFrameAnalyzer()
analyzer.visit(tree)

# Output the results.
print("RestrictedDataFrame variables found:", analyzer.restricted_dfs)
print("RestrictedDataFrame variables that used forbidden:", analyzer.forbidden_usage)