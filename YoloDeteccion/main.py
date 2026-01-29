import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8

model = YOLO('Models/detect_person.pt')  # Cambia al path de tu modelo

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(1)  # Usa 0 para la cámara por defecto

if not cap.isOpened():
    print("Error: No se puede acceder a la cámara.")
    exit()

print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el frame de la cámara.")
        break

    # Realizar la detección
    results = model(frame)  # El modelo toma el frame como entrada

    # Dibujar las cajas delimitadoras en la imagen
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extraer las coordenadas y la confianza
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas
            conf = box.conf[0]  # Confianza
            cls = box.cls[0]  # Clase detectada

            # Dibujar la caja y la etiqueta si la clase es 'persona' (cambia según tu modelo)
            if cls == 0:  # Suponiendo que la clase 'persona' es 0
                label = f"Persona: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el video con las detecciones
    cv2.imshow("Detección de Personas", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
