import sqlite3


def _extract(cur, lat, lon, r):
    cur.execute(f"""SELECT activity_ptr_id, title, rating, price, type, description, open_hours_id, service_options, types, ( 6371 * acos( cos( radians({lat}) ) * cos( radians( latitude ) ) * 
cos( radians( longitude ) - radians({lon}) ) + sin( radians({lat}) ) * 
sin( radians( latitude ) ) ) ) AS distance FROM itinerary_location GROUP BY activity_ptr_id HAVING
distance < {r} ORDER BY distance LIMIT 0 , 20;""")
    return cur.fetchall()


def get_hours_text(cur, id):
    cur.execute(f"SELECT monday, tuesday, wednesday, thursday, friday, saturday, sunday FROM itinerary_openhours WHERE id = {id}")
    hours = cur.fetchall()[0]
    return f"Monday: {hours[0]}, Tuesday: {hours[1]}, Wednesday: {hours[2]}, Thursday: {hours[3]}, Friday: {hours[4]}, Saturday: {hours[5]}, Sunday: {hours[6]}"


def get_text(cur, locations):
    response = ""
    for location in locations:
        hours = get_hours_text(cur, location[6])

        attributes = location[4] + "," + location[8] + ","

        for attribute in location[7].split(","):
            if len(attribute.split(":")) == 2:
                if attribute.split(":")[1] == "True":
                    attributes += attribute.split(":")[0] + ","
                else:
                    attributes += "no " + attribute.split(":")[0] + ","

        response += f"id: {location[0]}, name: {location[1]}, rating: {location[2]}, price: {location[3]}, description: {location[5]}, opening_hours: {hours}, attributes: {attributes}, distance: {location[9]}\n"

    return response


def get_locations(lat, lon):
    db_name = "whereigo.sqlite3"
    con = sqlite3.connect(db_name)
    cur = con.cursor()

    closest_locations = _extract(cur, lat, lon, 10)
    closest_locations_text = get_text(cur, closest_locations)
    con.close()
    return closest_locations_text


if __name__ == '__main__':
    lat = 45.0703
    lon = 7.6869
    print(get_locations(lat, lon))
