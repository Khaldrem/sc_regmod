import requests

BASE_API_URL = "https://www.yeastgenome.org/backend/"

def get_basic_info(locusId):
    r = requests.get(f"{BASE_API_URL}/locus/{locusId}")
    return r.json()


def get_position_info(locusId):
    r = requests.get(f"{BASE_API_URL}/locus/{locusId}/sequence_details")
    data = r.json()

    info = {}
    info["start"] = data["genomic_dna"][0]["start"]
    info["end"] = data["genomic_dna"][0]["end"]
    info["chromosomal_start"] = data["genomic_dna"][0]["tags"][0]["chromosomal_start"]
    info["chromosomal_end"] = data["genomic_dna"][0]["tags"][0]["chromosomal_end"]
    info["strand"] = data["genomic_dna"][0]["strand"]
    info["relative_start"] = data["genomic_dna"][0]["tags"][0]["relative_start"]
    info["relative_end"] = data["genomic_dna"][0]["tags"][0]["relative_end"]

    return info
    