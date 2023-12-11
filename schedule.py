import dill
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
from datetime import datetime
import pandas as pd


sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())
df = pd.read_csv('model/data/homework.csv')
with open(r'./model/pipeline.pkl', 'rb') as file:
    model = dill.load(file)


@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(frac=0.0005)
    data['predict_car_category'] = model['model'].predict(data)
    print(data[['id', 'price', 'predict_car_category']])


if __name__ == '__main__':
    sched.start()
