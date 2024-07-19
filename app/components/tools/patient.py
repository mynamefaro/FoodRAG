from sqlite3 import connect
from langchain_core import tools


class PatientRecordsTool:
    def __init__(self, datastore_url: str):
        self.__datastore_url = datastore_url
        self.conn = connect(self.__datastore_url)
        self.cursor = self.conn.cursor()
