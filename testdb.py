import pyodbc
import datetime

create_table_sql = """
                        CREATE TABLE ImageStats(
                            id int IDENTITY(1,1) CONSTRAINT pk_id PRIMARY KEY,
                            masked INTEGER NOT NULL,
                            date_updated DATETIME2 NOT NULL,
                        )
                   """

insert_data_sql =   """ INSERT INTO ImageStats (masked, date_updated) VALUES (?, ?)"""


conn = pyodbc.connect(Driver='{SQL Server};',
                      Server='DESKTOP-50CPBLJ\SQLEXPRESS;',
                      Database='facemask_db;',
                      Trusted_connection='yes;')

time_ = datetime.datetime.now()
to_insert = [0, time_]

cursor = conn.cursor()
cursor.execute(insert_data_sql, to_insert)
conn.commit()
print("Data inserted successfully")
  
conn.close()