import copy
import xml.etree.ElementTree as ET

def create(root, name):
  # Creates an element if it doesn't exist
  element = root.find(name)
  if element is None:
    root.append(ET.Element(name))

def copy_or_append(name, src, dst):
  element = src.find(name)
  if element is None:
    return

  if dst.find(name) is None:
    el = copy.deepcopy(element)
    wb = dst.find("worldbody")
    if wb is not None and name == "default":
      dst.insert(list(dst).index(wb), el)
    else:
      dst.append(el)
  else:
    dst.find(name).append(copy.deepcopy(element))

def copy_children(name, src, dst, exclude=None):

  # Check if there is something to copy
  elements = src.find(name)

  if elements is not None:

    # Create an element if necessary
    create(dst, name)

    # Copy each element except ones that are excluded
    for element in elements:
      if exclude is not None and \
          element.tag == exclude["tag"] and element.attrib[exclude["attrib"]] == exclude["name"]:
        continue
      else:
        dst.find(name).append(element)