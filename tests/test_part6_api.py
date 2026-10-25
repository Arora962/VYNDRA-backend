import os
from pathlib import Path
import pytest
@pytest.fixture()
def client(tmp_path, monkeypatch):
 monkeypatch.setenv('VYNDRA_DB_PATH',str(tmp_path/'test.sqlite3'))
 import vyndra.database as db
 db.DB_PATH=Path(os.environ['VYNDRA_DB_PATH'])
 from app import app
 app.config.update(TESTING=True)
 with app.test_client() as c: yield c

def test_health(client):
 r=client.get('/api/health'); assert r.status_code==200; assert r.json['status']=='ok'
def test_requires_json(client):
 r=client.post('/api/meals',data='not json',content_type='text/plain'); assert r.status_code==415
def test_meal_validation(client):
 assert client.post('/api/meals',json={}).status_code==400
 assert client.post('/api/meals',json={'items':[{'food_name':'apple','quantity_g':0}]}).status_code==400
def test_meal_unknown_food_does_not_guess(client):
 r=client.post('/api/meals',json={'items':[{'food_name':'not a real food name','quantity_g':100}]})
 assert r.status_code in (422,503)
def test_privacy_route(client):
 r=client.get('/api/privacy'); assert r.status_code==200; assert r.json['raw_chat_text_stored'] is False
