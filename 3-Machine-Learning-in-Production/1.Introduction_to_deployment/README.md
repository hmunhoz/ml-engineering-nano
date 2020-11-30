## What's ahead
In this lesson, you're going to get familiar with what's meant by _machine learning deployment_.

Specifically, we will look at answering the following questions:

1. What's the machine learning workflow?
2. How does deployment fit into the machine learning workflow?
3. What is cloud computing?
4. Why would we use cloud computing for deploying machine learning models?
5. Why isn't deployment a part of many machine learning curriculums?
6. What does it mean for a model to be deployed?
7. What are the essential characteristics associated with the code of deployed models?
8. What are different cloud computing platforms we might use to deploy our machine learning models?

## Machine Learning Workflow
![Machine Learning Workflow](./images/ml_workflow.png)
Deployment is the last part of a development cycle, where we make our model available to a software or a web application.


## Cloud Computing
### What is cloud computing?
Transforming an _Information Technology (IT) product_ into a _service_.

### Why would a business decide to use cloud computing?
Most of the factors related to choosing _cloud computing services_, instead of developing _on-premise IT resources_ are related to **_time_** and **_cost_**.

![IT Capacity](./images/capacity.png)
#### Benefits
1. Reduced Investments and Proportional Costs (providing cost reduction)
2. Increased Scalability (providing simplified capacity planning)
3. Increased Availability and Reliability (providing organizational agility)
#### Risks
1. (Potential) Increase in Security Vulnerabilities
2. Reduced Operational Governance Control (over cloud resources)
3. Limited Portability Between Cloud Providers
4. Multi-regional Compliance and Legal Issues
#### Other resources
- [National Institute of Standards and Technology](https://www.nist.gov/) formal definition of [Cloud Computing](https://csrc.nist.gov/publications/detail/sp/800-145/final).
- [Amazon Web Services](https://aws.amazon.com/)(AWS) discusses their definition of [Cloud Computing](https://aws.amazon.com/what-is-cloud-computing/)
- [Google Cloud Platform](https://cloud.google.com/)(GCP) discusses their definition of [Cloud Computing](https://cloud.google.com/what-is-cloud-computing/)
- [Microsoft Azure](https://azure.microsoft.com/en-us/)(Azure) discusses their definition of [Cloud Computing](https://azure.microsoft.com/en-us/overview/what-is-cloud-computing/)

### Machine Learning Applications
For **_personal use_**, one’s _likely_ to use cloud services, if they don’t have enough computing capacity.
With **_academic use_**, quite often one will use the university’s on-premise computing resources, given their availability. For smaller universities or research groups with few funding resources, _cloud services_ might offer a viable alternative to university computing resources.
For *_workplace usage_*, the amount of _cloud resources_ used depends upon an organization’s existing infrastructure and their vulnerability to the risks of cloud computing. A workplace may have security concerns, operational governance concerns, and/or compliance and legal concerns regarding _cloud usage_. Additionally, a workplace may already have on-premise infrastructure that supports the workflow; therefore, making _cloud usage_ an unnecessary expenditure. Keep in mind, many progressive companies may be incorporating _cloud computing_ into their business due to the business drivers and benefits of cloud computing.

### Paths to Deployment
#### Deployment to Production
**Recall that:**
**_Deployment to production_** can simply be thought of as a method that integrates a machine learning model into an existing production environment so that the model can be used to make _decisions_ or _predictions_ based upon _data input_ into the model.
#### Paths to deployment
From **_least_** to **_most_** _commonly_ used:
1. Python model is _recoded_ into the programming language of the production environment: Usually rewrite Python model to Java or C++ (for example). Rarely used because it takes time to recode, test and validate the model that provides the _same_ predictions as the _original_.
2. Model is _coded_ in _Predictive Model Markup Language_ (PMML) or _Portable Format Analytics_ (PFA): These are two complementary standards that _simplify_ moving predictive models to _deployment_ into a _production environment_. The Data Mining Group developed both PMML and PFA to provide vendor-neutral executable model specifications for certain predictive models used by data mining and machine learning. Certain analytic software allow for the direct import of PMML, such as IBM SPSS, R, SAS Base & Enterprise Miner, Apache Spark, Teradata Warehouse Miner, and TIBCO Spotfire.
3. Python model is _converted_ into a format that can be used in the production environment: _use libraries_ and _methods_ that _convert_ the model into **_code_** that can be used in the _production environment_. Most popular ML software frameworks (e.g. PyTorch, SciKit-Learn, TensorFlow) have methods that convert Python models into _intermediate standard format_, like ONNX ([Open Neural Network Exchange](https://onnx.ai/) format). This intermediate format can be converted into the software native to the production environment.
  - This is the _easiest_ and _fastest_ way **_to move_** a Python model from _modeling_ directly to _deployment_.
  - Moving forward, this is _tipically_ the way _models_ are **_moved_** into the _production environment_
  - Technologies like _containers_, _endpoints_, and _APIs_ also help **_ease_** the **_work_** required for _deploying_ a model into the _production environment_.
The **_third_** _method_ that's _most_ similar to what’s used for _deployment_ within **_Amazon’s SageMaker_**.

### Production Environments
![Production Environment](./images/production_enviromnent.png)

### Rest APIs
#### Model, Application, and Endpoint
![Model, Application, and Endpoint](./images/model_application_endpoint.png)
One way to think of the **_endpoint_** that acts as this _interface_, is to think of a _Python program_ where:
- the **endpoint** itself is like a **_function call_**
- the **_function_** itself would be the **model** and
- the **_Python program_** is the **application**.

#### Endpoint and REST API
Communication between the **application** and the **model** is done through the **endpoint** (_interface_), where the **endpoint** is an **Application Programming Interface** (**API**).
- An easy way to think of an **API**, is as a set of rules that enable programs, here the **application** and the **model**, to _communicate_ with each other.
- In this case, our **API** uses a **RE**presentational **S**tate **T**ransfer, **REST**, architecture that provides a framework for the _set of rules_ and _constraints_ that must be adhered to for _communication_ between programs.
- This **REST API** is one that uses _HTTP requests_ and _responses_ to enable communication between the **application** and the **model** through the **endpoint** (_interface_).
- Noting that _both_ the **HTTP request** and **HTTP response** are _communications_ sent between the **application** and **model**.

The **HTTP request** that is sent from your **application** to your **model** is composed of _four_ parts:
- **Endpoint**:
  - This **endpoint** will be in the form of a URL, Uniform Resource Locator, which is commonly know as a web address.
- HTTP Method:
  - There are _four_ main **HTTP methods**, but for purposes of **_deployment_** our **application** will use the **_POST method_** _only_.
    - **GET**: _READ_. This request is used to retrieve information. If the information is found, it is sent back as the response.
    - **POST**: _CREATE_. This request is used to create new information. Once a new entry is created, it tis sent back as the response.
    - **PUT**: _UPDATE_. This request is used to update information. The PATCH method also updates information, but it is only a partial update with PATCH.
    - **DELETE**: _DELETE_. This request is used to delete information.

The **HTTP response** sent from your model to your application is composed of _three_ parts:
- HTTP Status Code
  - If the model succesfully received and processed the _user's data_ that was sent in the **message**, the status code should start with a **_2_**, like _200_.
  - HTTP Headers
    - The **headers** will contain additional information, like the format of the data within the **message**, that's passed to the receiving program.
  - Message (Data or Body)
    - What's returned as the _data_ within the **message** is the _prediction_ that's provided by the **model**.

This _prediction_ is then presented to the _application user_ through the **application**. The **endpoint** is the **_interface_** that _enables communication_ between the **application** and the **model** using a **REST API**.
As we learn more about **REST****_ful_** **API**, realize that it's the **application’s** responsibility:
  - To format the _user’s data_ in a way that can be easily put into the **HTTP request** _message_ and _used_ by the **model**.
  - To translate the _predictions_ from the **HTTP response** _message_ in a way that’s easy for the _application user’s_ to understand.
Notice the following regarding the information included in the **_HTTP messages_** sent between **application** and **model**:
  - Often _user's data_ will need to be in a _CSV_ or _JSON_ format with a specific _ordering_ of the data that's dependent upon the **model** used.
  - Often _predictions_ will be returned in _CSV_ or _JSON_ format with a specific _ordering_ of the returned _predictions_ dependent upon the **model** used.

### Containers
