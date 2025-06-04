import json
import os
from datetime import datetime
from typing import List, Dict

LEADS_FILE = os.path.join(os.path.dirname(__file__), '..', 'leads.json')
os.makedirs(os.path.dirname(LEADS_FILE), exist_ok=True)


def _load() -> List[Dict]:
    if not os.path.exists(LEADS_FILE):
        return []
    with open(LEADS_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save(data: List[Dict]):
    with open(LEADS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_lead(name: str, email: str, interest: str):
    leads = _load()
    leads.append({
        'name': name,
        'email': email,
        'interest': interest,
        'timestamp': datetime.utcnow().isoformat()
    })
    _save(leads)


def list_leads() -> List[Dict]:
    return _load()
