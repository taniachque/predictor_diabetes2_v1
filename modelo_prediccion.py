import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



# Función de membresía difusa. SI FUCNIONA BIEN EN CONSOLA
def mareos(valor):
    if valor <= 0:
        return 0
    elif valor <= 1:
        return 0.25
    elif valor <= 2:
        return 0.5
    elif valor <= 3:
        return 0.75
    else:
        return 1

# Función de membresía para el conjunto difuso "visión borrosa"
def vision_borrosa(valor):
    if valor <= 0:
        return 0
    elif valor <= 1:
        return 0.25
    elif valor <= 2:
        return 0.5
    elif valor <= 3:
        return 0.75
    else:
        return 1

# Función de membresía para el conjunto difuso "fatiga"
def fatiga(valor):
    if valor <= 0:
        return 0
    elif valor <= 1:
        return 0.25
    elif valor <= 2:
        return 0.5
    elif valor <= 3:
        return 0.75
    else:
        return 1

# Función de membresía para el conjunto difuso "dolor de cabeza"
def dolor_cabeza(valor):
    if valor <= 0:
        return 0
    elif valor <= 1:
        return 0.25
    elif valor <= 2:
        return 0.5
    elif valor <= 3:
        return 0.75
    else:
        return 1
    
# Función de membresía para el conjunto difuso "dolor de pecho"
def dolor_pecho(valor):
    if valor <= 0:
        return 0
    elif valor <= 1:
        return 0.25
    elif valor <= 2:
        return 0.5
    elif valor <= 3:
        return 0.75
    else:
        return 1
    
# Función de membresía para el conjunto difuso "palpitaciones"
def palpitaciones(valor):
    if valor <= 0:
        return 0
    elif valor <= 1:
        return 0.25
    elif valor <= 2:
        return 0.5
    elif valor <= 3:
        return 0.75
    else:
        return 1

# Función de membresía para el conjunto difuso "dificultad para respirar"
def dificultad_respirar(valor):
    if valor <= 0:
        return 0
    elif valor <= 1:
        return 0.25
    elif valor <= 2:
        return 0.5
    elif valor <= 3:
        return 0.75
    else:
        return 1


# Función para combinar las inferencias utilizando el operador máximo
def combinar_inferencias(inferencias):
    return max(inferencias)

# Función para realizar la defusificación utilizando el método del centroide
def defusificar(inferencias):
    valores = [float(i) * float(j) for i, j in inferencias]
    suma_valores = sum(valores)
    suma_inferencias = sum([float(i) for i, _ in inferencias])
    if suma_inferencias != 0:
        return suma_valores / suma_inferencias
    else:
        return 0





# Cargar el dataset Pima  ........... :)...ESTE FUNCIONA BIEN//N pide presion rterail y da resadyo de presion, imc y peso y predice.
df = pd.read_csv('diabetes.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Solicitar los datos del paciente desde la consola
pregnancies = int(input("Número de embarazos: "))
glucose = int(input("Nivel de glucosa en plasma: "))

blood_pressure_input = input("Presión arterial: ")
if int(blood_pressure_input) > 0:
    blood_pressure = int(blood_pressure_input)

    # Asignar valores por defecto a las características desconocidas
    skin_thickness = 0
    insulin = 20

    # Verificar si el paciente proporcionó el valor del pliegue cutáneo
    skin_thickness_input = input("Espesor del pliegue cutáneo del tríceps (si desconoce, presione Enter): ")
    if skin_thickness_input:
        skin_thickness = int(skin_thickness_input)

    # Verificar si el paciente proporcionó el valor de la insulina
    insulin_input = input("Insulina en suero de 2 horas (si desconoce, presione Enter): ")
    if insulin_input:
        insulin = int(insulin_input)

    # Verificar si el paciente proporcionó el valor del pedigrí de diabetes
    diabetes_pedigree_input = input("¿Tiene algún familiar cercano diagnosticado con diabetes? (Ingrese 1 si sí, 0 si no): ")
    if diabetes_pedigree_input:
        diabetes_pedigree = float(diabetes_pedigree_input)

    # Calcular el BMI si el paciente proporcionó el peso y la talla
    bmi = 0.0
    weight_input = input("Peso en kg (si desconoce, presione Enter): ")
    height_input = input("Talla en metros (si desconoce, presione Enter): ")
    if weight_input and height_input:
        weight = float(weight_input)
        height = float(height_input)
        bmi = weight / (height ** 2)

    age = int(input("Ingrese su edad en años (CUMPLIDOS, NO POR CUMPLIR): "))


else:
    # Solicitar los valores de los síntomas
    mareos_val = float(input("Introduce el valor de mareos (0-4): "))
    vision_borrosa_val = float(input("Introduce el valor de visión borrosa (0-4): "))
    fatiga_val = float(input("Introduce el valor de fatiga (0-4): "))
    dolor_cabeza_val = float(input("Introduce el valor de dolor de cabeza (0-4): "))
    dolor_pecho_val = float(input("Introduce el valor de dolor en el pecho (0-4): "))
    palpitaciones_val = float(input("Introduce el valor de palpitaciones (0-4): "))
    dificultad_respirar_val = float(input("Introduce el valor de dificultad para respirar (0-4): "))

    # Evaluación de los grados de pertenencia
    mareos_pert = mareos(mareos_val)
    vision_borrosa_pert = vision_borrosa(vision_borrosa_val)
    fatiga_pert = fatiga(fatiga_val)
    dolor_cabeza_pert = dolor_cabeza(dolor_cabeza_val)
    dolor_pecho_pert = dolor_pecho(dolor_pecho_val)
    palpitaciones_pert = palpitaciones(palpitaciones_val)
    dificultad_respirar_pert = dificultad_respirar(dificultad_respirar_val)

    # Evaluación de las reglas difusas
    inferencias_hipotension = [
        mareos_pert,
        vision_borrosa_pert,
        fatiga_pert,
        dolor_cabeza_pert
    ]

    inferencias_hipertension = [
        dolor_cabeza_pert,
        vision_borrosa_pert,
        dolor_pecho_pert,
        palpitaciones_pert,
        dificultad_respirar_pert
    ]

    # Combinar las inferencias
    inferencia_hipotension = combinar_inferencias(inferencias_hipotension)
    inferencia_hipertension = combinar_inferencias(inferencias_hipertension)


    # Combinar las inferencias
    inferencia_hipotension = combinar_inferencias(inferencias_hipotension)
    inferencia_hipertension = combinar_inferencias(inferencias_hipertension)

    if all(inferencia == 0 for inferencia in inferencias_hipertension):
        resultado_hipertension = 0

    # Realizar la defusificación
    resultado_hipotension = defusificar([(inferencia_hipotension, 0)])
    resultado_hipertension = defusificar([(inferencia_hipertension, 1)])

    #TEST de resultado sintomas
    # Asignar el valor resultante a blood_pressure_input
    blood_pressure_input = str(resultado_hipertension)

    # Convertir el valor de blood_pressure_input a un formato adecuado para la presión arterial
    if resultado_hipertension == 0:
        blood_pressure = "80"
    elif resultado_hipertension >= 0.9:
        blood_pressure = "90"



    # Mostrar el resultado final
    print("Diagnóstico difuso:")
    #print("Hipotensión:", resultado_hipotension)

    if resultado_hipertension == 0:
        print("El paciente no tiene hipertensión. Su presión arterial se encuentra entre 110/70 y 120/80 mmHg.")
    elif resultado_hipertension >= 0.9:
        print("Hipertensión: Grado 1. Su presión arterial está alrededor de 140/90 mmHg")
    else:
        print("El paciente no tiene hipertensión")

    # Test
    # Asignar valores por defecto a las características desconocidas
    skin_thickness = 0
    insulin = 20

    # Verificar si el paciente proporcionó el valor del pliegue cutáneo
    skin_thickness_input = input("Espesor del pliegue cutáneo del tríceps (si desconoce, presione Enter): ")
    if skin_thickness_input:
        skin_thickness = int(skin_thickness_input)

    # Verificar si el paciente proporcionó el valor de la insulina
    insulin_input = input("Insulina en suero de 2 horas (si desconoce, presione Enter): ")
    if insulin_input:
        insulin = int(insulin_input)

    # Verificar si el paciente proporcionó el valor del pedigrí de diabetes
    diabetes_pedigree_input = input("¿Tiene algún familiar cercano diagnosticado con diabetes? (Ingrese 1 si sí, 0 si no): ")
    if diabetes_pedigree_input:
        diabetes_pedigree = float(diabetes_pedigree_input)

    # Calcular el BMI si el paciente proporcionó el peso y la talla
    bmi = 0.0
    weight_input = input("Peso en kg (si desconoce, presione Enter): ")
    height_input = input("Talla en metros (si desconoce, presione Enter): ")
    if weight_input and height_input:
        weight = float(weight_input)
        height = float(height_input)
        bmi = weight / (height ** 2)

    age = int(input("Ingrese su edad en años (CUMPLIDOS, NO POR CUMPLIR): "))



# Crear un nuevo DataFrame con los datos del paciente
new_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
                        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Unir los datos del paciente con el conjunto de prueba
X_test_with_patient = pd.concat([X_test, new_data])

# Entrenar el modelo de Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Realizar la predicción para el paciente
prediction = model.predict(X_test_with_patient.tail(1))
probabilities = model.predict_proba(X_test_with_patient.tail(1))

# Imprimir la predicción y probabilidades
if prediction[0] == 0:
    print("No se predice diabetes mellitus tipo 2 para el paciente.")
else:
    print("Se predice diabetes mellitus tipo 2 para el paciente.")

print("Probabilidad de tener diabetes mellitus tipo 2: {:.2f}".format(probabilities[0][1]))

# Determinar la categoría de peso según el BMI
if bmi < 18.5:
    weight_category = "Bajo peso"
elif bmi < 25:
    weight_category = "Peso normal"
elif bmi < 30:
    weight_category = "Sobrepeso"
else:
    weight_category = "Obesidad"


# Determinar la categoría de presión arterial
if int(blood_pressure) < 90:
    blood_pressure_category = "Hipotensión"
elif int(blood_pressure) < 120:
    blood_pressure_category = "Presión normal"
elif int(blood_pressure) < 140:
    blood_pressure_category = "Hipertensión arterial grado 1."
else:
    blood_pressure_category = "Hipertensión Arterial grado 2."

print("Categoría de presión arterial: {}".format(blood_pressure_category))


print("Categoría de peso según el BMI: {}".format(weight_category))

# Calcular y mostrar el accuracy del modelo
accuracy = model.score(X_test, y_test)
print("Accuracy del modelo: {:.2f}".format(accuracy))