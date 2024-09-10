import csv

file = open('ricette.csv', 'r')
csvreader = csv.reader(file)

next(csvreader)

for row in csvreader:
    name = row[0]
    category = row[1]
    main_ingredient = row[2]
    n_people = row[3]
    tags = {}
    if row[4] != "-":
        for tag in row[4].split("."):
            fields = tag.strip().split(":")
            if len(fields) == 2:
                tags[fields[0].strip()] = fields[1].strip()

    ingredients = {}
    for ingredient in row[5].split(r"\r\n"):
        fields = ingredient.split("====")
        if len(fields) == 2:
            if fields[0].strip() != "":
                ingredients[fields[1].strip()] = fields[0].strip()
            else:
                ingredients[fields[1].strip()] = "q.b."
        if len(fields) == 1:
            ingredients[fields[0].strip()] = "q.b."

    preparation = row[6]
    print(name, category, main_ingredient, n_people, tags, ingredients, preparation)
