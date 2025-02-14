import csv
import statistics
import os
def analyze_csv(input_filepath, output_filepath):
    """
    Lit un fichier CSV, calcule des statistiques descriptives pour chaque colonne numérique,
    et écrit les résultats dans un nouveau fichier CSV.

    Args:
        input_filepath: Chemin d'accès au fichier CSV d'entrée.
        output_filepath: Chemin d'accès au fichier CSV de sortie.
    """

    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Le fichier d'entrée '{input_filepath}' n'existe pas.")

    try:
        with open(input_filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile) # Use DictReader for easier access
            header = reader.fieldnames
            data = [row for row in reader]

        results = [
        for col_name in header:
            try:
                col_data = [float(row[col_name]) for row in data if row[col_name]]
                if col_data:
                mean = statistics.mean(col_data)
                median = statistics.median(col_data)
                    stdev = statistics.stdev(col_data) if len(col_data) > 1 else 0
                total = sum(col_data)
                    results.append([col_name, mean, median, stdev, total)
            except ValueError:
                print(f"La colonne '{col_name}' contient des données non numériques et sera ignorée.")

        # Écrire les résultats dans un nouveau fichier CSV
        with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Colonne', 'Moyenne', 'Médiane', 'Écart type', 'Somme'])
            writer.writerows(results)
        print(f"Les résultats ont été écrits dans '{output_filepath}'.")

    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
                results.append([col_name, mean, median, stdev, total])
                    results.append([col_name, mean, median, stdev, total)
            except ValueError:
                print(f"La colonne '{col_name}' contient des données non numériques et sera ignorée.")

        # Écrire les résultats dans un nouveau fichier CSV
        with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['Colonne', 'Moyenne', 'Médiane', 'Écart type', 'Somme'])
            writer.writerows(results)

        print(f"Les résultats ont été écrits dans '{output_filepath}'.")

    except Exception as e:
        print(f"Une erreur s'est produite: {e}")


# Exemple d'utilisation :
input_file = 'input.csv'
output_file = 'output.csv'

# Créer un fichier d'exemple (commentez si vous avez déjà un fichier input.csv)
with open(input_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Nom', 'Age', 'Note'])
    writer.writerow(['Alice', '25', '85'])
    writer.writerow(['Bob', '30', '92'])
    writer.writerow(['Charlie', '28', '78'])
    writer.writerow(['David', '', '88']) # Ligne avec une valeur manquante

analyze_csv(input_file, output_file)
