import xml.etree.ElementTree as ET
tree = ET.parse('CrossRec.xmi')
root = tree.getroot()


with open('train_model.txt','w',encoding='utf-8', errors='ignore') as res:

    for elem in root.iter():
        for sub in elem.iter():
            print(sub.tag, sub.attrib)


            for key, value in sub.attrib.items():
                print(key)
                print('-'*50)
                print(value)
                res.write(key+'#' + value+ ' '+"\n")


