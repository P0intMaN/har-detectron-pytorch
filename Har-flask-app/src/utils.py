import cv2

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)


def draw_line(image, p1, p2, color):
    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness=2, lineType=cv2.LINE_AA)


def find_person_indicies(scores):
    return [i for i, s in enumerate(scores) if s > 0.9]


def filter_persons(outputs):
    persons = {}
    p_indicies = find_person_indicies(outputs["instances"].scores)
    for x in p_indicies:
        desired_kp = outputs["instances"].pred_keypoints[x][:].to("cpu")
        persons[x] = desired_kp
    return persons, p_indicies


def draw_keypoints(person, img):
    l_eye = person[1]
    r_eye = person[2]
    l_ear = person[3]
    r_ear = person[4]
    nose = person[0]
    l_shoulder = person[5]
    r_shoulder = person[6]
    l_elbow = person[7]
    r_elbow = person[8]
    l_wrist = person[9]
    r_wrist = person[10]
    l_hip = person[11]
    r_hip = person[12]
    l_knee = person[13]
    r_knee = person[14]
    l_ankle = person[15]
    r_ankle = person[16]

    draw_line(img, l_shoulder, l_elbow, GREEN_COLOR)
    draw_line(img, l_elbow, l_wrist, GREEN_COLOR)
    draw_line(img, l_shoulder, r_shoulder, GREEN_COLOR)
    draw_line(img, l_shoulder, l_hip, GREEN_COLOR)
    draw_line(img, r_shoulder, r_hip, GREEN_COLOR)
    draw_line(img, r_shoulder, r_elbow, GREEN_COLOR)
    draw_line(img, r_elbow, r_wrist, GREEN_COLOR)
    draw_line(img, l_hip, r_hip, GREEN_COLOR)
    draw_line(img, l_hip, l_knee, GREEN_COLOR)
    draw_line(img, l_knee, l_ankle, GREEN_COLOR)
    draw_line(img, r_hip, r_knee, GREEN_COLOR)
    draw_line(img, r_knee, r_ankle, GREEN_COLOR)

    cv2.circle(img, (int(l_eye[0]), int(l_eye[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(r_eye[0]), int(r_eye[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(l_wrist[0]), int(l_wrist[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(r_wrist[0]), int(r_wrist[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(l_shoulder[0]), int(l_shoulder[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(r_shoulder[0]), int(r_shoulder[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(l_elbow[0]), int(l_elbow[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(r_elbow[0]), int(r_elbow[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(l_hip[0]), int(l_hip[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(r_hip[0]), int(r_hip[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(l_knee[0]), int(l_knee[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(r_knee[0]), int(r_knee[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(l_ankle[0]), int(l_ankle[1])), 4, WHITE_COLOR, -1)
    cv2.circle(img, (int(r_ankle[0]), int(r_ankle[1])), 4, WHITE_COLOR, -1)
