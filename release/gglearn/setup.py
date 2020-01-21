#!/usr/bin/env python
#-*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name = "gglearn",
    version = "0.1.4",
    keywords = ("learning", "analyze","data", "learn", "gg"),
    description = "Machine learning and data analysis library for Python 3",
    long_description = "Machine learning and data analysis library for Python 3！",
    license = "MIT Licence",

    url = "https://github.com/Wchenguang/gglearn/tree/master/release/gglearn",
    author = "Mr_W1997",
    author_email = "mr_w1997@163.com",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ['numpy', 'pandas', 'sklearn',
                        'matplotlib']
)
