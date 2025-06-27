import sqlite3
import face_recognition
import numpy as np
from ollie_print import olliePrint_simple

class PersonaService:
    def __init__(self, db_path="persona_memory.db"):
        self.db_path = db_path
        self._init_db()
        self.known_embeddings = []
        self.known_ids = []
        self.load_known_faces()
        olliePrint_simple("Persona service ready.")

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create personas table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            # Create embeddings table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                persona_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (persona_id) REFERENCES personas (id)
            )
            """)
            conn.commit()

    def load_known_faces(self):
        """Load all known face embeddings from the database into memory."""
        self.known_embeddings = []
        self.known_ids = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT persona_id, embedding FROM embeddings")
            for persona_id, embedding_blob in cursor.fetchall():
                self.known_ids.append(persona_id)
                self.known_embeddings.append(np.frombuffer(embedding_blob, dtype=np.float64))
        olliePrint_simple(f"Loaded {len(self.known_embeddings)} known face embeddings.")

    def get_persona_name(self, persona_id):
        """Retrieve a persona's name by their ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM personas WHERE id = ?", (persona_id,))
            result = cursor.fetchone()
            return result[0] if result else "Unknown"

    def recognize_faces(self, frame_array):
        """
        Recognize faces in a given frame and return their names and locations.
        This is the primary method to be called by the VisionService.
        """
        face_locations = face_recognition.face_locations(frame_array)
        face_embeddings = face_recognition.face_encodings(frame_array, face_locations)

        recognized_faces = []
        for i, face_embedding in enumerate(face_embeddings):
            if not self.known_embeddings:
                recognized_faces.append({"name": "An unknown person", "location": face_locations[i]})
                continue

            matches = face_recognition.compare_faces(self.known_embeddings, face_embedding, tolerance=0.6)
            face_distances = face_recognition.face_distance(self.known_embeddings, face_embedding)
            
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                persona_id = self.known_ids[best_match_index]
                name = self.get_persona_name(persona_id)
                recognized_faces.append({"name": name, "location": face_locations[i]})
                
                # Auto-improvement logic with quality-based capping (max 5 embeddings per persona)
                if face_distances[best_match_index] < 0.5: # High confidence match
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM embeddings WHERE persona_id = ?", (persona_id,))
                        count = cursor.fetchone()[0]
                        
                        if count >= 5: # At capacity - check if new embedding is better than worst existing
                            cursor.execute("""
                                SELECT id, embedding FROM embeddings 
                                WHERE persona_id = ? 
                                ORDER BY (
                                    SELECT MAX(face_distance(embedding, ?)) 
                                    FROM embeddings AS e2 
                                    WHERE e2.persona_id = ?
                                ) DESC
                                LIMIT 1
                            """, (persona_id, face_embedding.tobytes(), persona_id))
                            worst_id, worst_embedding = cursor.fetchone()
                            worst_embedding = np.frombuffer(worst_embedding, dtype=np.float64)
                            
                            if face_recognition.face_distance([worst_embedding], face_embedding)[0] > 0.3: # New is better
                                cursor.execute("DELETE FROM embeddings WHERE id = ?", (worst_id,))
                                conn.commit()
                                self._add_embedding_to_db(persona_id, face_embedding, conn)
                        else: # Under capacity - just add
                            self._add_embedding_if_novel(persona_id, face_embedding)
                        
                        self.load_known_faces() # Reload cache
            else:
                recognized_faces.append({"name": "An unknown person", "location": face_locations[i]})
        
        return recognized_faces

    def _add_embedding_if_novel(self, persona_id, new_embedding):
        """Adds a new embedding if it's sufficiently different from existing ones for that person."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings WHERE persona_id = ?", (persona_id,))
            
            is_novel = True
            for (embedding_blob,) in cursor.fetchall():
                existing_embedding = np.frombuffer(embedding_blob, dtype=np.float64)
                distance = face_recognition.face_distance([existing_embedding], new_embedding)
                if distance < 0.3: # If it's very similar to an existing one, it's not novel
                    is_novel = False
                    break
            
            if is_novel:
                olliePrint_simple(f"Adding novel embedding for persona ID {persona_id}.")
                self._add_embedding_to_db(persona_id, new_embedding, conn)
                self.load_known_faces() # Reload cache

    def enroll_person(self, frame_array, name):
        """Enroll a new person from a frame."""
        face_locations = face_recognition.face_locations(frame_array)
        if not face_locations:
            olliePrint_simple("No faces found in the enrollment frame.", "warning")
            return "No face found."
        if len(face_locations) > 1:
            olliePrint_simple("Multiple faces found. Please ensure only one person is in the frame for enrollment.", "warning")
            return "Multiple faces found."

        face_embedding = face_recognition.face_encodings(frame_array, face_locations)[0]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                # Create a new persona
                cursor.execute("INSERT INTO personas (name) VALUES (?)", (name,))
                persona_id = cursor.lastrowid
                
                # Add their first embedding
                self._add_embedding_to_db(persona_id, face_embedding, conn)
                conn.commit()
                
                self.load_known_faces() # Reload cache
                olliePrint_simple(f"Successfully enrolled {name} (ID: {persona_id}).", "success")
                return f"Enrolled {name} successfully."
            except sqlite3.IntegrityError:
                olliePrint_simple(f"A person named {name} already exists.", "warning")
                return f"{name} already exists."

    def _add_embedding_to_db(self, persona_id, embedding, conn):
        """Internal function to insert an embedding blob into the DB."""
        cursor = conn.cursor()
        cursor.execute("INSERT INTO embeddings (persona_id, embedding) VALUES (?, ?)",
                       (persona_id, embedding.tobytes()))

# Global persona service instance
persona_service = PersonaService() 