from peewee import *
import os

class DataBase:
    def __init__(self, db_path='record.db'):
        self.engine = SqliteDatabase(db_path)
        self.init_models()
        
        if not os.path.exists(db_path):
            self.create_tables()
        
    def connect(self):
        self.engine.connect()
    
    def create_tables(self):
        self.engine.create_tables([
            self.GuidedBench,
            self.JailbreakResponses,
            self.Evaluations
        ])

    def __del__(self):
        self.engine.close()
        
    def init_models(self):
        class GuidedBench(Model):
            index = AutoField(primary_key=True)

            topic = CharField()
            question = CharField()

            e_d1 = CharField()
            e_d2 = CharField()
            e_d3 = CharField()

            f_d1 = CharField()
            f_d2 = CharField()
            f_d3 = CharField()

            e_e1 = CharField()
            e_e2 = CharField()
            e_e3 = CharField()

            f_e1 = CharField()
            f_e2 = CharField()
            f_e3 = CharField()

            target = CharField()

            class Meta:
                database = self.engine
                table_name = 'guided_bench'

        class JailbreakResponses(Model):
            index = ForeignKeyField(GuidedBench, backref='jailbreak_responses')
            method = CharField()
            victim = CharField()
            prompt = TextField()
            response = TextField()
            tag = TextField() # for other information

            class Meta:
                database = self.engine
                table_name = 'jailbreak_responses'
                primary_key = CompositeKey('index', 'method', 'victim')

        class Evaluations(Model):
            index = ForeignKeyField(GuidedBench, backref='evaluations')
            method = CharField()
            victim = CharField()
            scoring = CharField()
            evaluator = CharField()
            value = FloatField()
            reason = TextField()
            tag = TextField() # for other information

            class Meta:
                database = self.engine
                table_name = 'evaluations'
                primary_key = CompositeKey('index', 'method', 'victim', 'scoring', 'evaluator')

        self.GuidedBench = GuidedBench
        self.JailbreakResponses = JailbreakResponses
        self.Evaluations = Evaluations