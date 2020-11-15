## Software Engineering Practices

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
- Follow **PEP 8** https://www.python.org/dev/peps/pep-0008/?#code-lay-out

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
- **PEP 257**: https://www.python.org/dev/peps/pep-0257/
- **NumPy Docstrings**: https://numpydoc.readthedocs.io/en/latest/format.html

### Version Control in Data Science

- How to Version Control Your Production Machine Learning Models: https://blog.algorithmia.com/how-to-version-control-your-production-machine-learning-models/
- Versioning Data Science: https://shuaiw.github.io/2017/07/30/versioning-data-science.html


