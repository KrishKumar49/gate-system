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
    
    if not (face_visit and vehicle_visit and plate_visit):
        return False, 0.0, None
    
    if not (face_visit == vehicle_visit == plate_visit):
        return False, 0.0, None
    
    

    if face_score is None:
        face_score = 0.0
    if vehicle_score is None:
        vehicle_score = 0.0
    if plate_score is None:
        plate_score = 0.0
    
    # Weighted average with more emphasis on face recognition
    total_score = (face_score * 0.5) + (vehicle_score * 0.3) + (plate_score * 0.2)
    
    if total_score >= 0.85:
        return True, total_score, face_visit
    
    elif face_score >= 0.9 and vehicle_score >= 0.8:
        return True, total_score, face_visit
    
    else:
        return False, total_score, face_visit