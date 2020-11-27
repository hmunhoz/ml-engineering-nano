## Introduction to Object-Oriented Programming

### Procedural vs Object-Oriented Programming
- Procedural: "List of actions"
- Object: In OOP, we model things as objects. Think of objects as things that exist in the real world. These objects are composed by two main things:
  - Characteristics (nouns) &rarr; _attributes_
  - Actions (verbs) &rarr; _methods_

### Class, object, method, attribute
- Class: a _recipe_ for creating an object, consisting methods and attributes. We can have a class that defines a shirt (color, size, model, etc.)
- Object: an specific instance of a class. In our case, a defined shirt (yellow, medium size, short sleeve, etc.). We could have another shirt object (instance) based on the same class, but with different characteristics (green, small size, long sleeve, etc.)
- Method: an action that a class or object could take.
- Attribute: a descriptor, or characteristic.
- Encapsulation: combine functions and data all into a single entity. In object-oriented programming, this single entity is called a class.

### OOP Syntax
```class ClassName:
  def __init__(self, *args, **kwargs):
    self.arg1 = ...
```
- Always have the ```__init__``` method. This will initialize/instantiate your class;
- The ```self``` parameter serves the purpose of differentiating different _objects_ of the same _class_;

### A couple of notes about OOP

#### Set and Get methods
In python, we can change an object's attribute like this:
  ```
  shirt_one.color = 'red'
  ```
This, however, is not a good practice in general OOP. Instead, we could create methods to change or retrieve
attribute values for objects. These are called ```Set``` and ```Get``` methods.
  ```
  class Shirt:
    def __init__(self, price):
      self._price = price

    # Getter
    def get_price(self):
      return self._price

    # Setter
    def set_price(self, new_price):
      self._price = new_price
  ```
In python, there is a convention that states that attributes that starts with an underscore should not
be accessed directly, e.g: ```self._price```.

#### A note about attributes
There are a few drawbacks to accessing attributes directly instead of writing a method to access such attributes.
Changing values via a method gives you more flexibility in the long-term. What if the units of measurement change, like the store was originally meant to work in US dollars and now has to handle Euros?
If you changed manually throughout your code, you would have to manually update everything to work with Euros. If you, however, have used methods, all you would have to do is update the method code.

### Magic Methods
Magic methods are a way of overriding and customizing some default behavior for python operations. For example, we could edit the ```__add__``` method to correctly add two Gaussian curves together.

### Inheritance
Inheritance helps organize code with a more general version of a class (parent) and then specific children. Updates made to a parent class automatically trickles down to its children, and can make object-oriented code more efficient to write.

### Advanced OOP Topics
We've been exposed to:
- classes and objects
- attributes and methods
- magic methods
- inheritance

But there are other interesting topics:
- [class methods, instance methods and static methods](https://realpython.com/instance-class-and-static-methods-demystified/) - these are different types of methods that can be accessed at the class or object level
- [class attributes vs instance attributes](https://www.python-course.eu/python3_class_and_instance_attributes.php) - you can also define attributes at the class level or at the instance level
- [multiple inheritance, mixins](https://easyaspython.com/mixins-for-fun-and-profit-cb9962760556) - A class can inherit from multiple parent classes
- [Python decorators](https://realpython.com/primer-on-python-decorators/) - Decorators are a short-hand way for using functions inside other functions
