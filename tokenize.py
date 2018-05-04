import ast
import re

import astor
from nltk.tokenize import RegexpTokenizer
import spacy


EN = spacy.load('en')


def tokenize_docstring(text):
    tokens = EN.tokenizer(text)
    return [token.text.lower() for token in tokens if not token.is_space]


def tokenize_code(text):
    return RegexpTokenizer(r'\w+').tokenize(text)


def get_function_docstring_pairs(blob):
    pairs = []
    try:
        module = ast.parse(blob)
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        for _class in classes:
            functions.extend([node for node in _class.body if isinstance(node, ast.FunctionDef)])

        for f in functions:
            if ast.get_docstring(f):
                source = astor.to_source(f)
                function = astor.to_source(f).replace(ast.get_docstring(f, clean=False), '')
                docstring = ast.get_docstring(f)
                if function and docstring:
                    pairs.append((source,
                                  ' '.join(tokenize_code(function)),
                                  ' '.join(tokenize_docstring(docstring.split('\n\n')[0])),
                                  f.name,
                                  f.lineno))
    except (SyntaxError, MemoryError, UnicodeEncodeError):
        continue
    return pairs
