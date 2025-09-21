from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

#Banco de dados
def init_db():
    conn = sqlite3.connect("carros.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS carros (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        marca TEXT,
        modelo TEXT,
        ano INTEGER,
        preco REAL
    )
    """)
    conn.commit()
    conn.close()

def query_db(query, args=(), one=False):
    conn = sqlite3.connect("carros.db")
    cursor = conn.cursor()
    cursor.execute(query, args)
    conn.commit()
    result = cursor.fetchall()
    conn.close()
    return (result[0] if result else None) if one else result

#Rotas
@app.route("/cars", methods=["POST"])
def add_car():
    data = request.json
    query_db("INSERT INTO carros (marca, modelo, ano, preco) VALUES (?,?,?,?)",
             (data["marca"], data["modelo"], data["ano"], data["preco"]))
    return jsonify({"message": "Carro adicionado com sucesso"})

@app.route("/cars", methods=["GET"])
def get_cars():
    rows = query_db("SELECT * FROM carros")
    return jsonify(rows)

@app.route("/cars/<int:car_id>", methods=["GET"])
def get_car(car_id):
    row = query_db("SELECT * FROM carros WHERE id=?", (car_id,), one=True)
    return jsonify(row if row else {"message": "Carro não encontrado"})


@app.route("/cars/<int:car_id>", methods=["PUT"])
def update_car(car_id):
    data = request.json
    query_db("UPDATE carros SET marca=?, modelo=?, ano=?, preco=? WHERE id=?",
             (data["marca"], data["modelo"], data["ano"], data["preco"], car_id))
    return jsonify({"message": "Carro atualizado com sucesso"})


@app.route("/cars/<int:car_id>", methods=["DELETE"])
def delete_car(car_id):
    query_db("DELETE FROM carros WHERE id=?", (car_id,))
    return jsonify({"message": "Carro deletado com sucesso"})

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
