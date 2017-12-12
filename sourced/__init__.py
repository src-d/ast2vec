# You must not include any other code and data in a namespace package's __init__.py
import pkg_resources
pkg_resources.declare_namespace(__name__)
