"""Flask API blueprint for nutrition, meal logging, chat, and privacy-safe persistence."""
from __future__ import annotations
import os
from flask import Blueprint, jsonify, request
from .database import init_db, save_meal, get_meal, list_meals, save_chat_event
from .nutrition import find_food, calculate_nutrition, get_food_table
bp=Blueprint('vyndra_api', __name__, url_prefix='/api')
MAX_JSON_BYTES=64*1024

def _json_object():
 if request.content_length is not None and request.content_length > MAX_JSON_BYTES:
  return None, (jsonify({'error':'Request body too large; maximum is 64 KB.'}),413)
 if not request.is_json: return None,(jsonify({'error':'Content-Type must be application/json.'}),415)
 payload=request.get_json(silent=True)
 if not isinstance(payload,dict): return None,(jsonify({'error':'Body must be a JSON object.'}),400)
 return payload,None

@bp.get('/health')
def health():
 try: init_db(); return jsonify({'status':'ok','service':'vyndra-backend','database':'sqlite'})
 except Exception: return jsonify({'status':'error','error':'Database initialization failed.'}),500

@bp.get('/foods/search')
def foods_search():
 q=(request.args.get('q') or '').strip()
 if not q: return jsonify({'error':'Query parameter q is required.'}),400
 if len(q)>100: return jsonify({'error':'q must be 100 characters or fewer.'}),400
 try:
  df=get_food_table(); hits=df[df['food_name'].str.contains(q,case=False,regex=False,na=False)].head(20)
  return jsonify({'query':q,'count':len(hits),'foods':hits['food_name'].astype(str).tolist()})
 except Exception: return jsonify({'error':'Nutrition database is unavailable. Check data/raw/Db.csv.'}),503

@bp.get('/nutrition/<path:food_name>')
def nutrition_lookup(food_name):
 if not food_name.strip() or len(food_name)>150: return jsonify({'error':'food_name must be 1–150 characters.'}),400
 try: return jsonify(find_food(food_name))
 except FileNotFoundError: return jsonify({'error':'Nutrition database file is missing.'}),503
 except Exception: return jsonify({'error':'Nutrition lookup failed.'}),500

@bp.post('/nutrition/calculate')
def nutrition_calculate():
 p,e=_json_object()
 if e:return e
 name=p.get('food_name'); grams=p.get('grams')
 if not isinstance(name,str) or not name.strip() or len(name)>150: return jsonify({'error':'food_name must be a non-empty string up to 150 characters.'}),400
 if isinstance(grams,bool) or not isinstance(grams,(int,float)) or not 0 < grams <= 5000: return jsonify({'error':'grams must be a number greater than 0 and at most 5000.'}),400
 try:return jsonify(calculate_nutrition(name,float(grams)))
 except FileNotFoundError:return jsonify({'error':'Nutrition database file is missing.'}),503
 except ValueError as ex:return jsonify({'error':str(ex)}),400

@bp.post('/meals')
def create_meal():
 p,e=_json_object()
 if e:return e
 raw=p.get('items')
 if not isinstance(raw,list) or not 1<=len(raw)<=30:return jsonify({'error':'items must be a list containing 1–30 foods.'}),400
 items=[]
 for idx,item in enumerate(raw):
  if not isinstance(item,dict):return jsonify({'error':f'items[{idx}] must be an object.'}),400
  name=item.get('food_name'); grams=item.get('quantity_g',100)
  if not isinstance(name,str) or not name.strip() or len(name)>150:return jsonify({'error':f'items[{idx}].food_name must be a non-empty string up to 150 characters.'}),400
  if isinstance(grams,bool) or not isinstance(grams,(int,float)) or not 0 < grams <= 5000:return jsonify({'error':f'items[{idx}].quantity_g must be > 0 and <= 5000.'}),400
  try: n=calculate_nutrition(name,float(grams))
  except FileNotFoundError:return jsonify({'error':'Nutrition database file is missing.'}),503
  if not n.get('found'):return jsonify({'error':f'No exact nutrition match for {name!r}; correct the name rather than guessing nutrition values.'}),422
  items.append({'food_name':n['food_name'],'quantity_g':float(grams),'calories':n['energy_kcal'],'protein_g':n['protein_g'],'carbs_g':n['carb_g'],'fat_g':n['fat_g'],'fiber_g':n['fibre_g'],'match_type':'exact_normalized_name'})
 notes=p.get('notes','')
 if not isinstance(notes,str) or len(notes)>500:return jsonify({'error':'notes must be a string of at most 500 characters.'}),400
 try: return jsonify(save_meal(items,source='manual',notes=notes)),201
 except Exception:return jsonify({'error':'Could not save meal to SQLite.'}),500

@bp.get('/meals')
def meals_list():
 raw=request.args.get('limit','20')
 try: limit=int(raw)
 except ValueError:return jsonify({'error':'limit must be an integer between 1 and 100.'}),400
 if not 1<=limit<=100:return jsonify({'error':'limit must be between 1 and 100.'}),400
 try:return jsonify({'meals':list_meals(limit)})
 except Exception:return jsonify({'error':'Could not read meals from SQLite.'}),500

@bp.get('/meals/<int:meal_id>')
def meal_detail(meal_id):
 meal=get_meal(meal_id)
 return (jsonify(meal),200) if meal else (jsonify({'error':'Meal not found.'}),404)

@bp.post('/chat/analyze')
def chat_analyze():
 p,e=_json_object()
 if e:return e
 text=p.get('text')
 if not isinstance(text,str) or not text.strip():return jsonify({'error':'text must be a non-empty string.'}),400
 if len(text)>4000:return jsonify({'error':'text must be 4000 characters or fewer.'}),413
 try:
  from .chat_intelligence import analyze_sentiment, extract_food_mentions, predict_food_intent
  intent=predict_food_intent(text); mentions=extract_food_mentions(text); sentiment=analyze_sentiment(text)
  # Store derived labels and a one-way hash only, never raw chat text.
  save_chat_event(str(intent),str(sentiment),mentions,text)
  return jsonify({'food_intent':intent,'food_mentions':mentions,'sentiment':sentiment,'privacy':{'raw_text_persisted':False,'stored_fields':['derived labels','food mentions','text length','SHA-256 digest']}})
 except Exception:return jsonify({'error':'Chat analysis is unavailable; check Part 4 dependencies and demo data.'}),503

@bp.get('/chat/topics')
def chat_topics():
 try:
  from .chat_intelligence import DEFAULT_CHAT_DATA, _read_csv, run_topic_model
  frame=_read_csv(DEFAULT_CHAT_DATA)
  return jsonify(run_topic_model(frame['text'].astype(str).tolist()))
 except Exception:return jsonify({'error':'Topic analysis is unavailable; check Part 4 demo data and dependencies.'}),503

@bp.get('/privacy')
def privacy():
 return jsonify({'raw_chat_text_stored':False,'image_retention':'Images are not retained by the Part 6 API by default.','database':'SQLite local file; protect it as user data.','limits':{'chat_characters':4000,'meal_items':30,'food_name_characters':150},'note':'No authentication is implemented. Do not expose this development API publicly without authentication, HTTPS, rate limiting, and a retention policy.'})
