import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import OneHotEncoder

def prepare_data(df, dataset_type='train'):
    def fix_distance_km(val):
        if pd.isna(val) or val == '' or val is None:
            return 1.0
        try:
            val_str = str(val).replace(',', '').strip().lower()
            match = re.search(r'(\d+(\.\d+)?)', val_str)
            if not match:
                return 1.0
            num = float(match.group(1))
            if 'ק"מ' in val_str or 'km' in val_str:
                return num
            if 'מטר' in val_str or 'meter' in val_str or 'm' in val_str:
                return num / 1000
            return num / 1000 if num > 20 else num
        except:
            return 1.0

    def fix_floor_values(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).strip().lower()
        if any(word in val_str for word in ["קרקע", "ground", "basement", "מרתף"]):
            return 0.0
        if "מרתף" in val_str or "basement" in val_str:
            match = re.search(r'-?\d+', val_str)
            if match:
                return float(match.group())
        match = re.search(r'\d+', val_str)
        if match:
            return float(match.group())
        return 0.0

    is_train = dataset_type == 'train'
    df = df.copy()

    df['distance_from_center'] = df['distance_from_center'].apply(fix_distance_km)
    df['floor'] = df['floor'].apply(fix_floor_values)
    df['garden_area'] = df['garden_area'].fillna(0)
    df['monthly_arnona'] = pd.to_numeric(df['monthly_arnona'], errors='coerce')
    df['building_tax'] = pd.to_numeric(df['building_tax'], errors='coerce')

    df['monthly_arnona'] = df['monthly_arnona'].fillna(df.groupby(['neighborhood', 'room_num'])['monthly_arnona'].transform('mean')
                              ).fillna(df.groupby('neighborhood')['monthly_arnona'].transform('mean')
                              ).fillna(df['monthly_arnona'].mean())
    df['building_tax'] = df['building_tax'].fillna(df.groupby(['neighborhood', 'room_num'])['building_tax'].transform('mean')
                              ).fillna(df.groupby('neighborhood')['building_tax'].transform('mean')
                              ).fillna(df['building_tax'].mean())

    if is_train:
        if 'price' in df.columns:
            df = df[(df['price'] >= 1000) & (df['price'] <= 100000)]
        if 'area' in df.columns:
            df = df[df['area'] <= 1000]
        if 'garden_area' in df.columns:
            df = df[df['garden_area'] <= 500]
        if 'distance_from_center' in df.columns:
            df = df[df['distance_from_center'] <= 20000]
        if 'monthly_arnona' in df.columns:
            df = df[df['monthly_arnona'] <= 3000]
        if 'building_tax' in df.columns:
            df = df[df['building_tax'] <= 3000]

    if 'price' in df.columns and not is_train:
        df = df.drop(columns=['price'])

    if 'neighborhood' in df.columns:
        df['neighborhood'] = df['neighborhood'].astype(str).str.strip()
        def is_invalid_neighborhood(name):
            if not isinstance(name, str): return True
            name = name.strip()
            if name in ['כללי', 'אחר', 'unknown', '', 'nan', 'NaN']: return True
            if re.fullmatch(r'[A-Za-z\s\-]+', name): return True
            return False
        if is_train:
            df = df[~df['neighborhood'].apply(is_invalid_neighborhood)]

    numeric_cols = ['room_num', 'floor', 'area', 'garden_area',
                    'num_of_payments', 'monthly_arnona', 'building_tax',
                    'total_floors', 'num_of_images']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if all(col in df.columns for col in ['monthly_arnona', 'area', 'neighborhood']):
        df['arnona_per_sqm'] = df['monthly_arnona'] / df['area']
        avg_arnona_per_sqm = df.groupby('neighborhood')['arnona_per_sqm'].mean()
        def fill_arnona(row):
            if pd.isna(row['monthly_arnona']):
                avg = avg_arnona_per_sqm.get(row['neighborhood'], None)
                return avg * row['area'] if avg else row['monthly_arnona']
            return row['monthly_arnona']
        df['monthly_arnona'] = df.apply(fill_arnona, axis=1)
        df.drop(columns='arnona_per_sqm', inplace=True)

    if is_train:
        medians = {col: df[col].median() for col in numeric_cols}
        with open('feature_medians.pkl', 'wb') as f:
            pickle.dump(medians, f)
    else:
        with open('feature_medians.pkl', 'rb') as f:
            medians = pickle.load(f)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0))

    def enrich_description(df):
        keywords = {
            "has_parking": "חניה",
            "has_storage": "מחסן",
            "has_safe_room": "ממ\"ד",
            "has_balcony": "מרפסת",
            "is_renovated": "משופץ",
            "is_furnished": "מרוהט"
        }
        if 'description' in df.columns:
            for col, keyword in keywords.items():
                if col not in df.columns or df[col].isna().all():
                    df[col] = df['description'].str.contains(keyword, na=False).astype(int)
        return df
    df = enrich_description(df)

    binary_cols = ['has_parking', 'has_storage', 'elevator', 'ac', 'handicap',
                   'has_bars', 'has_safe_room', 'has_balcony', 'is_furnished', 'is_renovated']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'כן': 1, 'לא': 0, 'Yes': 1, 'No': 0,
                                   'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(int)

    df['garden_area'] = df.get('garden_area', 0).fillna(0)
    if 'room_num' in df.columns and 'area' in df.columns:
        df['area_per_room'] = np.where(df['room_num'] > 0, df['area'] / df['room_num'], df['area'])

    def floor_score(row):
        if pd.isna(row['floor']) or pd.isna(row['total_floors']): return 1
        if row['floor'] == 0: return 2
        elif row['floor'] == row['total_floors']: return 1.5
        else: return 1
    df['floor_score'] = df.apply(floor_score, axis=1)

    if 'neighborhood' in df.columns:
        if is_train and 'price' in df.columns:
            hood_avg = df.groupby('neighborhood')['price'].mean().to_dict()
            with open('neighborhood_prices.pkl', 'wb') as f:
                pickle.dump(hood_avg, f)
        else:
            try:
                with open('neighborhood_prices.pkl', 'rb') as f:
                    hood_avg = pickle.load(f)
            except:
                hood_avg = {}
        general_avg = np.mean(list(hood_avg.values())) if hood_avg else 5000
        df['neighborhood_price_level'] = df['neighborhood'].map(hood_avg).fillna(general_avg)

    if 'property_type' in df.columns:
        df['property_type'] = df['property_type'].astype(str).str.strip()
    df['property_type'] = df['property_type'].replace(['nan', 'None', 'NaN'], np.nan)
    df['property_type'] = df['property_type'].fillna('דירה')
    df = df[~df['property_type'].isin(['חניה', 'מחסן'])]

    property_mapping = {
        'דירה': 'דירה',
        'דירה להשכרה': 'דירה',
        'דירת גן': 'דירת גן',
        'דירת גן להשכרה': 'דירת גן',
        'דופלקס': 'דופלקס',
        'יחידת דיור': 'יחידת דיור',
        'גג/פנטהאוז': 'גג/ פנטהאוז',
        'גג/ פנטהאוז': 'גג/ פנטהאוז',
        'גג/פנטהאוז להשכרה': 'גג/ פנטהאוז',
        'דו משפחתי': 'דו משפחתי',
        'מרתף/פרטר': 'מרתף/פרטר',
        'סטודיו/לופט': 'סטודיו/ לופט',
        'סאבלט': 'סאבלט',
        'החלפת דירות': 'החלפת דירות',
        "פרטי/קוטג'": "בית פרטי/ קוטג'",
        "בית פרטי/ קוטג'": "בית פרטי/ קוטג'",
        'Квартира': 'דירה'
    }
    df['property_type'] = df['property_type'].map(property_mapping).fillna('כללי')

    cat_cols = ['property_type']
    df[cat_cols] = df[cat_cols].astype(str).fillna("missing")

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = ohe.fit_transform(df[cat_cols])

    encoded_df = pd.DataFrame(encoded, index=df.index, columns=ohe.get_feature_names_out(cat_cols))
    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    cols_to_drop = ['distance_from_center', 'days_to_enter', 'num_of_images']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    return df.select_dtypes(include=[np.number])
