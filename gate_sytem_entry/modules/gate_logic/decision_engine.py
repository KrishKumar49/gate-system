def calculate_gate_score(face_result, vehicle_result, plate_result):
    face_score = face_result.get("score", 0.0)
    vehicle_score = vehicle_result.get("score", 0.0)
    plate_score = plate_result.get("score", 0.0)
    
    face_visit = face_result.get("visit_id")
    vehicle_visit = vehicle_result.get("visit_id")
    plate_visit = plate_result.get("visit_id")
    
    print(
        face_visit, 
        vehicle_visit,
        plate_visit,
    )
    
    valid_visits = [v for v in [face_visit, vehicle_visit, plate_visit] if v is not None]
    
    if not valid_visits:
        return False, 0.0, None
    
    if len(set(valid_visits)) > 1:
        return False, 0.0, None
    
    consensus_visit = valid_visits[0]
    
    face_s = face_score if face_result.get("verified") else 0.0
    vehicle_s = vehicle_score if vehicle_result.get("verified") else 0.0
    plate_s = plate_score if plate_result.get("verified") else 0.0
    
    total_score = (face_s * 0.5) + (vehicle_s * 0.3) + (plate_s * 0.2)
    
    if total_score >= 0.55:
        return True, total_score, consensus_visit
    
    if face_s >= 0.8 and plate_s >= 0.7:
        return True, total_score, consensus_visit
    
    return False, total_score, consensus_visit