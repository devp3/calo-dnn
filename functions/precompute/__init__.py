import logging
import os

function_list = [
    item for item in os.listdir('functions/precompute') 
    if item.endswith(".py") and item != "__init__.py"
    ]
function_list = [item[:-3] for item in function_list]

logging.debug("Importing functions.precompute.__init__.py")
logging.debug("List of functions in functions.precompute:")
for f in function_list:
    logging.debug(f)
logging.debug("Total number of functions: " + str(len(function_list)))

__all__ = function_list
