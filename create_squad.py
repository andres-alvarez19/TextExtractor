import json


def create_squad_entry(context, questions_answers, context_id):
    """
    Crea una entrada para el conjunto de datos SQuAD con IDs únicos.

    :param context: Texto del contexto.
    :param questions_answers: Diccionario donde la clave es la pregunta y el valor es una lista de respuestas.
    :param context_id: ID único para identificar el contexto.
    :return: Diccionario con el formato SQuAD.
    """
    qas = []
    for idx, (question, answers) in enumerate(questions_answers.items(), start=1):
        qas.append({
            "id": f"c{context_id}_q{idx}",
            "question": question,
            "answers": [{"text": ans["text"], "answer_start": ans["start"]} for ans in answers]
        })
    return {
        "context": context,
        "qas": qas
    }


def create_squad_dataset(entries):
    """
    Crea un conjunto de datos SQuAD a partir de múltiples entradas.

    :param entries: Lista de entradas SQuAD generadas con create_squad_entry.
    :return: Diccionario en formato SQuAD.
    """
    paragraphs = []
    for entry in entries:
        paragraphs.append(entry)
    return {"data": [{"title": "Custom Dataset", "paragraphs": paragraphs}]}


# Ejemplo de entradas con IDs únicos
entries = [
    create_squad_entry(
        context="The farm collected 30 manure samples, 15 feed samples, and 19 water samples. ESBL-Klebsiella spp. was studied in this research.",
        questions_answers={
            "How many manure samples were collected in this study?": [
                {"text": "30", "start": 19},
                {"text": "manure samples: 30", "start": 4}
            ],
            "Which bacteria was studied in this research?": [
                {"text": "ESBL-Klebsiella spp.", "start": 64},
                {"text": "Klebsiella spp.", "start": 69}
            ]
        },
        context_id=1
    ),
    create_squad_entry(
        context="The antibiotics ceftiofur and streptomycin were used in dairy calves to control infections.",
        questions_answers={
            "Which antibiotics were used in animals?": [
                {"text": "ceftiofur", "start": 15},
                {"text": "streptomycin", "start": 29}
            ],
            "Have antibiotics been used in animals?": [
                {"text": "Yes", "start": 0}
            ]
        },
        context_id=2
    ),
    create_squad_entry(
        context="Rectal fecal samples (n = 508) and manure, feed, and water samples (n = 64) were collected from 14 dairy farms in Tennessee.",
        questions_answers={
            "How many manure samples were collected in this study?": [
                {"text": "30", "start": 38},
                {"text": "manure samples: 30", "start": 17}
            ]
        },
        context_id=3
    ),
    create_squad_entry(
        context="Individual animal rectal fecal samples (dairy cows, n = 424; calves, n = 84) and farm environmental samples (manure, n = 30; feed, n = 15; and water, n = 19) were collected.",
        questions_answers={
            "How many manure samples were collected in this study?": [
                {"text": "30", "start": 98},
                {"text": "manure, n = 30", "start": 77}
            ]
        },
        context_id=4
    ),
    create_squad_entry(
        context="ESBL-Klebsiella spp. were recovered from all sample types, including 30 manure samples.",
        questions_answers={
            "How many manure samples were collected in this study?": [
                {"text": "30", "start": 54},
                {"text": "30 manure samples", "start": 43}
            ]
        },
        context_id=5
    ),
    create_squad_entry(
        context="A total of 572 samples were collected, of which 30 were manure samples.",
        questions_answers={
            "How many manure samples were collected in this study?": [
                {"text": "30", "start": 50},
                {"text": "30 were manure samples", "start": 43}
            ]
        },
        context_id=6
    )
]

# Crear el conjunto de datos SQuAD
squad_data = create_squad_dataset(entries)

# Guardar el conjunto de datos como JSON
output_file = "custom_squad.json"
with open(output_file, "w") as f:
    json.dump(squad_data, f, indent=2)

print(f"SQuAD dataset creado y guardado como '{output_file}'")
