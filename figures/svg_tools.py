import base64
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib as mpl

def math_font_css():
    styles=[]
    for name in ["cmmi10","cmr10","cmsy10"]:
        path=Path(mpl.get_data_path())/"fonts"/"ttf"/f"{name}.ttf"
        encoded=base64.b64encode(path.read_bytes()).decode()
        styles.append(f'@font-face{{font-family:"{name}";src:url(data:font/ttf;base64,{encoded}) format("truetype");font-weight:400}}')
    return "\n".join(styles)

def svg(path, prefix):
    root=ET.fromstring(path.read_text())
    ET.register_namespace("","http://www.w3.org/2000/svg")
    ET.register_namespace("xlink","http://www.w3.org/1999/xlink")
    ids={el.attrib["id"]:prefix+el.attrib["id"] for el in root.iter() if "id" in el.attrib}
    for el in root.iter():
        for name,value in list(el.attrib.items()):
            if name=="id":el.set(name,ids[value])
            else:
                for old,new in ids.items():
                    value=value.replace(f"url(#{old})",f"url(#{new})")
                    if value=="#"+old:value="#"+new
                el.set(name,value)
    root.set("role","img")
    root.set("aria-label","Conceptual decoding and intervention illustration")
    return ET.tostring(root,encoding="unicode")
