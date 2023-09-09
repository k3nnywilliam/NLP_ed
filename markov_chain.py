import random

class MarkovChainTextGenerator:
    def __init__(self, order=1):
        self.order = order
        self.model = {}

    def train(self, text):
        words = text.split()
        for i in range(len(words) - self.order):
            gram = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            if gram in self.model:
                self.model[gram].append(next_word)
            else:
                self.model[gram] = [next_word]

    def generate_text(self, length=50):
        current = random.choice(list(self.model.keys()))
        text = list(current)
        for _ in range(length - self.order):
            if current in self.model:
                next_word = random.choice(self.model[current])
                text.append(next_word)
                current = tuple(text[-self.order:])
            else:
                break
        return ' '.join(text)

# Example usage:
if __name__ == "__main__":
    source_text = "This is a sample text for Markov chain text generation. It demonstrates how Markov chains can be used to generate new text."

    generator = MarkovChainTextGenerator(order=2)
    generator.train(source_text)
    
    for i in range(10):
        generated_text = generator.generate_text(length=20)
        n = i+1
        print(f"Text {n} : {generated_text}")