from langchain_community.utilities import SQLDatabase


class DoctorSQL(SQLDatabase):
    def __init__(self):
        super().__init__()

    def get_doctor(self, doctor_id):
        query = f"SELECT * FROM doctor WHERE doctor_id = {doctor_id}"
        return self.execute_query(query)

    def get_doctors(self):
        query = "SELECT * FROM doctor"
        return self.execute_query(query)
