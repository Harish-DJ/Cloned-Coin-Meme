"""
static_content.py
─────────────────
Hardcoded curriculum content for the 10 topics across 3 difficulty levels.
Used to run the AI Tutor demo 100% locally without an LLM API.
"""

# Explanations for [topic_idx][diff_idx]
EXPLANATIONS = {
    0: { # Number Sense
        0: "Number sense is all about understanding how numbers work together. Think of it as the foundation of a house. Today we'll focus on the order of operations (PEMDAS). Always solve Parentheses first, then Exponents, Multiplication/Division, and finally Addition/Subtraction.",
        1: "Building on basic number sense, let's look at fractions and decimals. They are just different ways of representing parts of a whole. For example, 1/2 is exactly the same as 0.5. To add fractions, you always need a common denominator.",
        2: "Advanced number sense involves complex nested operations. When dealing with multiple brackets and negative fractions, precision is key. Work from the innermost absolute value or bracket outwards, applying negative signs carefully."
    },
    1: { # Algebra Basics
        0: "Algebra simply uses letters (like 'x') to represent unknown numbers. If I have 3 boxes of apples and 2 more boxes, I have 5 boxes total. In algebra, we write this as: 3x + 2x = 5x. We call 'x' a variable.",
        1: "When simplifying algebraic expressions, you can only combine 'like terms'. You cannot add an x to an x². Think of x as apples and x² as oranges. 3x + 2x² remains exactly as it is.",
        2: "Advanced simplification involves distributing negative signs across long polynomials. Remember that a negative outside a parenthesis flips the sign of every single term inside: -(2x - 3y + 4) becomes -2x + 3y - 4."
    },
    2: { # Linear Equations
        0: "A linear equation is like a balanced scale. Whatever you do to one side, you must do to the other to keep it balanced. To solve x + 5 = 12, subtract 5 from both sides to find x = 7.",
        1: "Two-step equations require peeling back layers to isolate 'x'. Deal with addition and subtraction first, then multiplication and division. For 2x - 4 = 10, first add 4, then divide by 2.",
        2: "When variables appear on both sides of the equals sign, your first goal is to gather them all onto one side. If 3x + 5 = x - 7, subtract x from both sides to get 2x + 5 = -7, then solve normally."
    },
    3: { # Systems of Equations
        0: "A system of equations is just two equations happening at the same time. The solution is the point where their lines cross on a graph. If y = 2x and y = 4, then 2x must equal 4, so x = 2.",
        1: "Substitution is a powerful method. If one equation tells you that y = x + 3, you can replace 'y' in the second equation with (x + 3) to solve for x.",
        2: "Elimination involves adding or subtracting two equations together to 'eliminate' one variable. Sometimes you need to multiply an entire equation by a number first to make the coefficients match up perfectly."
    },
    4: { # Quadratic Equations
        0: "A quadratic equation has an x². While linear equations make straight lines, quadratics make U-shapes called parabolas. The simplest quadratic is x² = 9, which has two answers: x=3 and x=-3.",
        1: "Factoring is a quick way to solve quadratics. You want to find two numbers that multiply to the last term and add to the middle term. For x² + 5x + 6 = 0, the numbers are 2 and 3, so (x+2)(x+3)=0.",
        2: "When a quadratic cannot be easily factored, we use the Quadratic Formula. It guarantees a solution. The formula uses the coefficients a, b, and c from the standard form ax² + bx + c = 0."
    },
    5: { # Polynomials
        0: "A polynomial is just a chain of terms added together, like x³ + 2x² - 5. The generic rule is to combine terms with the exact same exponent. Adding (x² + 2x) + (3x² + x) gives 4x² + 3x.",
        1: "Multiplying binomials requires the FOIL method: First, Outer, Inner, Last. For (x+2)(x+3), multiply the First terms (x²), Outer (3x), Inner (2x), and Last (6), then combine to get x² + 5x + 6.",
        2: "Polynomial long division is similar to numerical long division. Focus only on eliminating the leading term at each step. It is highly useful for finding complex roots of higher-degree polynomials."
    },
    6: { # Trigonometry
        0: "Trigonometry studies triangles. The three basic functions—Sine, Cosine, and Tangent—relate the angles of a right triangle to its sides. A helpful memory trick is SOH CAH TOA.",
        1: "The Unit Circle is a way to find trig functions for angles larger than 90 degrees. It has a radius of 1. For any angle, the x-coordinate on the circle is the Cosine, and the y-coordinate is the Sine.",
        2: "Trigonometric identities are equations that are always true. The most famous is the Pythagorean Identity: sin²(θ) + cos²(θ) = 1. This can be used to dramatically simplify complex expressions."
    },
    7: { # Statistics Basics
        0: "Statistics helps us make sense of data. The three main measures are Mean (the average), Median (the exact middle number when sorted), and Mode (the most common number).",
        1: "Standard Deviation tells us how spread out our data is. A low standard deviation means most numbers are very close to the average. A high standard deviation means the numbers are scattered widely.",
        2: "Z-scores tell you exactly how many standard deviations a data point is from the mean. It allows us to compare scores from entirely different tests by standardizing them into a single metric."
    },
    8: { # Probability
        0: "Probability is the chance that an event will happen, ranging from 0 (impossible) to 1 (certain). If you flip a fair coin, the probability of getting Heads is 1 out of 2, or 0.5.",
        1: "Independent events don't affect each other. If you flip a coin and roll a die, one doesn't change the other. To find the probability of both happening, you simply multiply their individual probabilities.",
        2: "Bayes' Theorem works with conditional probabilities. It allows you to update your probability estimate after receiving new evidence. It is foundational to machine learning algorithms."
    },
    9: { # Calculus Intro
        0: "Calculus is the mathematics of change. A derivative simply measures the 'rate of change' or the slope of a curve at a single, specific point, much like a speedometer in a car.",
        1: "To find the derivative of simple powers of x, use the Power Rule: bring the exponent down to the front, and reduce the exponent by 1. So the derivative of x³ is 3x².",
        2: "Integration is the opposite of derivation; it finds the total accumulated area under a curve. The Fundamental Theorem of Calculus links these two concepts together."
    }
}

# Questions for [topic_idx][diff_idx]
QUESTIONS = {
    0: {
        0: {"question": "What is 8 + (3 × 4)?", "options": ["44", "20", "15", "24"], "correct_index": 1, "explanation": "PEMDAS: Multiply 3×4=12 first, then add 8 to get 20."},
        1: {"question": "What is 1/2 + 1/4?", "options": ["2/6", "3/4", "1/8", "2/4"], "correct_index": 1, "explanation": "Convert 1/2 to 2/4. Then 2/4 + 1/4 = 3/4."},
        2: {"question": "Evaluate: -|-5| + (-3)²", "options": ["4", "-14", "14", "-4"], "correct_index": 0, "explanation": "Absolute value of -5 is 5. With the negative outside, it's -5. (-3)² is 9. So -5 + 9 = 4."}
    },
    1: {
        0: {"question": "Simplify: 4x + 3x", "options": ["7x²", "12x", "7x", "x"], "correct_index": 2, "explanation": "Like terms are added by their coefficients: 4 + 3 = 7."},
        1: {"question": "Simplify: 2x² + 3x - x² + 4", "options": ["4x² + 4", "x² + 3x + 4", "5x² + 4", "x² + 7"], "correct_index": 1, "explanation": "Combine x² terms: 2x² - x² = x². The 3x and 4 cannot be combined."},
        2: {"question": "Simplify: -(3x - 2y) + 4(x - y)", "options": ["x - 6y", "-x - 2y", "x - 2y", "7x - 6y"], "correct_index": 2, "explanation": "-3x + 2y + 4x - 4y. Combine x: 4x-3x=x. Combine y: 2y-4y=-2y. Result: x - 2y."}
    },
    2: {
        0: {"question": "Solve for x: x - 4 = 10", "options": ["x = 6", "x = 14", "x = 40", "x = -6"], "correct_index": 1, "explanation": "Add 4 to both sides: 10 + 4 = 14."},
        1: {"question": "Solve for x: 3x + 7 = 22", "options": ["x = 5", "x = 6", "x = 15", "x = 4"], "correct_index": 0, "explanation": "Subtract 7 to get 3x = 15. Then divide by 3 to get x = 5."},
        2: {"question": "Solve for x: 5x - 3 = 2x + 12", "options": ["x = 3", "x = 4", "x = 5", "x = 15"], "correct_index": 2, "explanation": "Subtract 2x: 3x - 3 = 12. Add 3: 3x = 15. Divide by 3: x = 5."}
    },
    3: {
        0: {"question": "If y = 5 and x + y = 12, what is x?", "options": ["x = 5", "x = 17", "x = 7", "x = 12"], "correct_index": 2, "explanation": "Substitute y: x + 5 = 12. Subtract 5 to get x = 7."},
        1: {"question": "Solve: y = 2x and x + y = 9", "options": ["x=3, y=6", "x=6, y=3", "x=4, y=5", "x=2, y=7"], "correct_index": 0, "explanation": "Substitute 2x for y: x + 2x = 9 -> 3x = 9 -> x = 3. Then y = 2(3) = 6."},
        2: {"question": "Solve by elimination: 2x + y = 7 and 3x - y = 8", "options": ["x=1, y=5", "x=3, y=1", "x=2, y=3", "x=4, y=-1"], "correct_index": 1, "explanation": "Add the equations: (2x+3x) + (y-y) = 7+8 -> 5x = 15 -> x=3. 2(3)+y=7 -> y=1."}
    },
    4: {
        0: {"question": "Solve: x² = 16", "options": ["x = 4", "x = -4", "x = 4 or -4", "x = 8 or -8"], "correct_index": 2, "explanation": "Both positive and negative 4 squared equal 16."},
        1: {"question": "Factor: x² + 7x + 10 = 0", "options": ["(x+5)(x+2)", "(x+10)(x+1)", "(x-5)(x-2)", "(x+7)(x+3)"], "correct_index": 0, "explanation": "5 and 2 multiply to 10 and add to 7."},
        2: {"question": "In the Quadratic Formula, what does the 'b² - 4ac' part determine?", "options": ["The slope", "Number of real roots", "The y-intercept", "The highest point"], "correct_index": 1, "explanation": "This is the discriminant. If positive, 2 roots. If zero, 1 root. If negative, 0 real roots."}
    },
    5: {
        0: {"question": "Add: (2x + 1) + (3x + 4)", "options": ["5x + 5", "6x + 4", "5x + 4", "6x² + 5"], "correct_index": 0, "explanation": "Combine x terms: 2x+3x=5x. Combine numbers: 1+4=5."},
        1: {"question": "Expand: (x + 3)(x + 4)", "options": ["x² + 12", "x² + 7x + 12", "x² + 7", "2x + 7"], "correct_index": 1, "explanation": "FOIL: x² + 4x + 3x + 12 = x² + 7x + 12."},
        2: {"question": "If you divide a polynomial of degree 4 by a polynomial of degree 1, what is the degree of the quotient?", "options": ["1", "4", "3", "5"], "correct_index": 2, "explanation": "Degree rules for division: subtract the degrees. 4 - 1 = 3."}
    },
    6: {
        0: {"question": "In a right triangle, Sine is:", "options": ["Adjacent/Hypotenuse", "Opposite/Hypotenuse", "Opposite/Adjacent", "Adjacent/Opposite"], "correct_index": 1, "explanation": "SOH: Sine = Opposite / Hypotenuse."},
        1: {"question": "On the Unit Circle, the coordinates (1, 0) correspond to what angle?", "options": ["90 degrees", "180 degrees", "0 degrees", "270 degrees"], "correct_index": 2, "explanation": "Angle 0 starts on the positive x-axis, at the point (1,0)."},
        2: {"question": "Simplify: 1 - sin²(x)", "options": ["sin²(x)", "cos²(x)", "tan²(x)", "1"], "correct_index": 1, "explanation": "From the Pythagorean identity: sin²(x) + cos²(x) = 1, therefore 1 - sin²(x) = cos²(x)."}
    },
    7: {
        0: {"question": "Find the Mean of: 2, 4, 6", "options": ["2", "4", "6", "12"], "correct_index": 1, "explanation": "(2 + 4 + 6) / 3 = 12 / 3 = 4."},
        1: {"question": "Find the Median of: 1, 9, 3, 5, 7", "options": ["3", "9", "5", "7"], "correct_index": 2, "explanation": "Sorted order: 1, 3, 5, 7, 9. The middle number is 5."},
        2: {"question": "If a dataset has a high standard deviation, its bell curve graph will look:", "options": ["Tall and narrow", "Short and wide", "Perfectly flat", "A straight vertical line"], "correct_index": 1, "explanation": "High spread means data is pushed outwards, flattening out the peak."}
    },
    8: {
        0: {"question": "What is the probability of rolling a '4' on a standard 6-sided die?", "options": ["1/4", "1/6", "4/6", "1/2"], "correct_index": 1, "explanation": "There is one '4' out of 6 possible sides."},
        1: {"question": "What is the probability of flipping Heads twice in a row?", "options": ["1/2", "1/4", "1/8", "1"], "correct_index": 1, "explanation": "1/2 chance for the first flip × 1/2 chance for the second flip = 1/4."},
        2: {"question": "A bag has 3 red and 2 blue marbles. You draw one red marble. What is the probability the NEXT marble is ALSO red?", "options": ["3/5", "2/5", "2/4 (or 1/2)", "1/5"], "correct_index": 2, "explanation": "After removing 1 red, the bag has 2 reds and 2 blues left, so 2 out of 4 total."}
    },
    9: {
        0: {"question": "The derivative represents the:", "options": ["Total area", "Y-intercept", "Slope", "Average"], "correct_index": 2, "explanation": "A derivative measures the instantaneous rate of change, or slope, at a specific point."},
        1: {"question": "What is the derivative of x²?", "options": ["x", "2x", "x³", "2"], "correct_index": 1, "explanation": "Power Rule: Bring the 2 down, reduce exponent by 1. The result is 2x¹ = 2x."},
        2: {"question": "What is the derivative of 3x⁴?", "options": ["12x³", "3x³", "4x³", "12x⁴"], "correct_index": 0, "explanation": "Multiply 4 by the coefficient 3 to get 12, then subtract 1 from the exponent: 12x³."}
    }
}

def get_static_explanation(topic_idx, diff_idx, student_name):
    base_text = EXPLANATIONS.get(topic_idx, {}).get(diff_idx, "Focus carefully on this topic.")
    return f"Hi {student_name}! {base_text} Let's dive in."

def get_static_question(topic_idx, diff_idx):
    return QUESTIONS.get(topic_idx, {}).get(diff_idx, {
        "question": "Static Fallback Question",
        "options": ["A", "B", "C", "D"],
        "correct_index": 0,
        "explanation": "Please update static curriculum."
    })
