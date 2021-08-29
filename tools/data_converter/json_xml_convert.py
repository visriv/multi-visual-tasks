import json
import xmltodict


# convert xml to json
def xml_to_json(xml_str):
    # xml parser
    xml_parse = xmltodict.parse(xml_str)
    # dumps() convert dict to json format, loads() convert json to dict format
    json_str = json.dumps(xml_parse, indent=1)
    return json_str


# convert json to xml
def json_to_xml(json_str):
    # using unparse() in xmltodict to convert json to xml
    xml_str = xmltodict.unparse(json_str, pretty=1)
    return xml_str


def test_main():

    b = """<?xml version="1.0" encoding="utf-8"?>
            <user_info>
                <id>12</id>
                <name>Tom</name>
                <age>12</age>
                <height>160</height>
                <score>100</score>
                <variance>12</variance>
            </user_info>
        """

    a = {
        "user_info": {
            "id": 12,
            "name": "Tom",
            "age": 12,
            "height": 160,
            "score": 100,
            "variance": 12,
        }
    }

    print("---------------------------split----------------------------------")
    print(xml_to_json(b))
    print("---------------------------split----------------------------------")
    print(json_to_xml(a))
    print("---------------------------split----------------------------------")


if __name__ == "__main__":
    test_main()
