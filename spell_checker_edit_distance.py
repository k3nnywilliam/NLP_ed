import editdistance

# Sample dictionary of known words
dictionary = {"shenanigans", "shannon", "burp", "searing", "atom"}

def spell_checker(input_word):
    # Calculate the edit distance between the input word and dictionary words
    distances = [(word, editdistance.eval(input_word, word)) for word in dictionary]

    # Sort the words by edit distance in ascending order
    suggestions = sorted(distances, key=lambda x: x[1])

    # Get the top N suggestions (you can adjust N as needed)
    top_suggestions = [word for word, distance in suggestions if distance <= 2]  # Consider words with distance <= 2

    return top_suggestions

if __name__ == "__main__":
    #user_input = input("Enter a word: ").strip().lower()
    word = "senaigans"
    suggestions = spell_checker(word)

    if word in dictionary:
        print(f"{word} is a valid word.")
    else:
        if suggestions:
            print(f"Did you mean one of these words? {', '.join(suggestions)}")
        else:
            print("No suggestions found.")

