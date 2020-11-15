## 1. Software Engineering Practices

### Clean and Modular Code
- PRODUCTION CODE: software running on production servers
- PRODUCTION _QUALITY_ CODE: code that meets expectations regarding reliability, scalability, efficiency, etc.
- CLEAN: readable, simple and concise code.
- MODULAR: logically separated into modules and functions. Makes code more organized, efficient and reusable.
- MODULE: a file. Modules allow code to be reused by encapsulating them into files that can be imported into other files.

### Refactoring Code

- *Reestructuring* your internal code without changing its external functionality. Used to _clean_ and _modularize_ your code after it is working.
- It is not easy to write the best code while creating a prototype. Refactoring is important and will make you a better developer in the long run.

### Writing Clean Code

#### Choosing good names

- **Be descriptive and imply type**: for booleans, use the prefix ```is_``	 or ```has_``` to make it clear it is a condition. You can use verbs for function names or nouns for variables.
- **Be consistent but clearly differentiate**: ```age_list``` and ```age``` is easier to differentiate than ```ages``` and ```age```.
- **Avoid single letters and abbreviations**: except (maybe) for counters and math expressions (e.g. x and y)
- **Long names != descriptive names**: You should be descriptive, but only with relevant information. E.g. good functions names describe what they do well without including details about implementation or highly specific uses.

#### Nice whitespace
- Consistent identation
- Separate sections with blank lines
- Limit lines to around 79 characters
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/?#code-lay-out)

### Writing Modular Code
- **DRY**: Don't Repeat Yourself - generalize and consolidate repeated code in functions or loops
- **Abstract out logic to improve readability**: Abstracting out code into a function not only makes it less repetitive, but also improves readability with descriptive function names. Be careful with over-engineering, though.
- **Minimize the number of entities (functions, classes, modules...)**: Be careful not to create too many modules and having to jump around code.
- **Functions should do ONE thing**: If a functions accumulates too many duties, it becomes harder to reuse it.
- **Arbitrary/General names could improve readability in some functions**: Try using ```arr``` as an array (list) representation, for example.
- **Try to use less than 3 arguments per function**: Makes easier to understand. Sometimes, we cannot avoid it.

### Efficient Code
- Run faster
- Take up less space
Try to optimize data types and vectorization of your code. Look for standard implementations first, instead of coming up with your own.


### Documentation

- Helpful for claryfying complex parts of your code.
- **In-line comments** - line level
- **Docstrings** -  module and function level
- **Project Documentation** - project level

#### In-Line Comments
- Explain major steps of complex code. Do not explain line by line, nor *what* the code is doing, but *why*. Be careful not to use it to justify bad code.

#### Docstrings
- Explain the functionality of a module or function. Ideally, every function should have a docstring.
- [PEP 257](https://www.python.org/dev/peps/pep-0257/)
- [NumPy Docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)

### Version Control in Data Science

- [How to Version Control Your Production Machine Learning Models](https://blog.algorithmia.com/how-to-version-control-your-production-machine-learning-models/)
- [Versioning Data Science](https://shuaiw.github.io/2017/07/30/versioning-data-science.html)

### Testing
Testing is essential before deployment. Prevents faulty conclusions or errors before they make a significant impact.

### Testing And Data Science
- Problems that occur in DS settings aren't always easy to identify, such as inappropriate usage of features, incorrect encoding, etc.
- In order to identify such errors, we should check for the quality and accuracy of both our *analysis* and *code*.
- **TEST DRIVEN DEVELOPMENT (TDD)**: Write tests for tasks before even writing the code to implement those tasks.
- **UNIT TEST**: Type of test that covers a "unit" of code, like a single function, independently of from the rest of the program
- Four Ways Data Science Goes Wrong and How Test Driven Data Analysis Can Help [Blog Post](https://www.predictiveanalyticsworld.com/patimes/four-ways-data-science-goes-wrong-and-how-test-driven-data-analysis-can-help/6947/)
- Ned Batchelder: Getting Started Testing: [Slide Deck](https://speakerdeck.com/pycon2014/getting-started-testing-by-ned-batchelder) and [Presentation Video](https://www.youtube.com/watch?v=FxSsnHeWQBY)

### Unit Tests
We'd like to test our functions in a repeatable and automated way.
#### Advantages and Disadvantages of Unit Testing
- **Advantages**: Unit Tests are isolated from the rest of our program, so no dependencies are involved. No access to databases, APIs or other external sources are required.
- **Disadvantages**: Not always sufficient to prove that our program is working. We need to test that the parts of our program work and interact properly with each other, communicating and transferring data between them.
- [Integration Testing](https://www.fullstackpython.com/integration-testing.html)

### Unit Testing Tools
- Use a library, such as ```pytest```. Getting Started [here](https://docs.pytest.org/en/latest/getting-started.html)
  - Create a test file starting with ```test_```
  - Define unit test functions starting with ```test_``` inside the test file
  - Enter ```pytest``` into the terminal in the directory of test file.
- Preferrably, use only one ```assert``` statement per test.

### Test Driven Development And Data Science
- **TDD**: Write tests before implementing the functions.
- Tests can check for different scenarios and edge cases. This way, we can see if our function passes all of the requirements while developing it.
- When refactoring or adding code, tests help to assure that the rest of the code won't break. Also assures repeatability of code, regardless of external parameters, such as hardware time.

#### TDD for Data Science
- [Data Science TDD](https://www.linkedin.com/pulse/data-science-test-driven-development-sam-savage/)
- [TDD for Data Science](http://engineering.pivotal.io/post/test-driven-development-for-data-science/)
- [TDD is Essential for Good Data Science Here's Why](https://medium.com/@karijdempsey/test-driven-development-is-essential-for-good-data-science-heres-why-db7975a03a44)
- [Testing Your Code](http://docs.python-guide.org/en/latest/writing/tests/) (general Python TDD)

### Logging
Valuable for understanding the events that occur while running your program.

### Log Messages
- **Be Professional and Clear**:
  - ```Bad: Hmmm... this isn't working???```
  - ```Bad: idk.... :(```
  - ```Good: Couldn't parse file.```
- **Be concise and use normal capitalization**:
  - ```Bad: Start Product Recommendation Process```
  - ```Bad: We have completed the steps necessary and will now proceed with the recommendation process for the records in our product database.```
  - ```Good: Generating product recommendations.```
- **Choose the appropriate level for logging**:
  - DEBUG - level you would use for anything that happens in the program.
  - ERROR - level to record any error that occurs
  - INFO - level to record all actions that are user-driven or system specific, such as regularly scheduled operations
- **Provide any useful information**:
  - ```Bad: Failed to read location data```
  - ```Good: Failed to read location data: store_id 8324971```


### Code Review

Benefits the whole team to promote good programming practices and prepare code for production.
- [Code Review](https://github.com/lyst/MakingLyst/tree/master/code-reviews)
- [Code Review Best Practices](https://www.kevinlondon.com/2015/05/05/code-review-best-practices.html)

### Questions to Ask Yourself When Conducting a Code Review

- **Is the code clean and modular?**
  - Can I understand the code easily?
  - Does it use meaningful names and whitespace?
  - Is there duplicated code?
  - Can you provide another layer of abstraction?
  - Is each function and module necessary?
  - Is each function or module too long?
- **Is the code efficient?**
  - Are there loops or other steps we can vectorize?
  - Can we use better data structures to optimize any steps?
  - Can we shorten the number of calculations needed for any steps?
  - Can we use generators or multiprocessing to optimize any steps?
- **Is documentation effective?**
  - Are in-line comments concise and meaningful?
  - Is there complex code that's missing documentation?
  - Do function use effective docstrings?
  - Is the necessary project documentation provided?
- **Is the code well tested?**
  - Does the code high test coverage?
  - Do tests check for interesting cases?
  - Are the tests readable?
  - Can the tests be made more efficient?
- **Is the logging effective?**
  - Are log messages clear, concise, and professional?
  - Do they include all relevant and useful information?
  - Do they use the appropriate logging level?
  
### Tips for Conduction a Code Review
- Use a code linter
- Explain issues and make suggestions
  - ```BAD: Make model evaluation code its own module - too repetitive.```
  - ```BETTER: Make the model evaluation code its own module. This will simplify models.py to be less repetitive and focus primarily on building models.```
  - ```GOOD: How about we consider making the model evaluation code its own module? This would simplify models.py to only include code for building models. Organizing these evaluations methods into separate functions would also allow us to reuse them with different models without repeating code.```
- Keep your comments objective and impersonal
  - ```BAD: I wouldn't groupby genre twice like you did here... Just compute it once and use that for your aggregations.```
  - ```BAD: You create this groupby dataframe twice here. Just compute it once, save it as groupby_genre and then use that to get your average prices and views.```
-   ```GOOD: Can we group by genre at the beginning of the function and then save that as a groupby object? We could then reference that object to get the average prices and views without computing groupby twice.```
- Provide code examples. Let's say we are reviewing this:
```
first_names = []
last_names = []

for name in enumerate(df.name):
    first, last = name.split(' ')
    first_names.append(first)
    last_names.append(last)

df['first_name'] = first_names
df['last_names'] = last_names
```
  - ```BAD: You can do this all in one step by using the pandas str.split method.```
  - ```GOOD: We can actually simplify this step to the line below using the pandas str.split method. Found this on this stack overflow post: https://stackoverflow.com/questions/14745022/how-to-split-a-column-into-two-columns df['first_name'], df['last_name'] = df['name'].str.split(' ', 1).str```
