from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import pandas as pd
import os


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = 'llama3-8b-8192'

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name=model
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/run_ml_task", response_class=HTMLResponse)
async def run_ml_task(request: Request, ml_problem: str = Form(...), file_upload: UploadFile = File(None)):
    try:
        if file_upload:
            # Read the file as bytes
            file_contents = await file_upload.read()
            
            # Attempt to read the file with different encodings
            df = pd.read_csv(file_contents, encoding='utf-8', error_bad_lines=False).head(5)
            data_upload = True
        else:
            data_upload = False
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": f"Error reading the file: {e}"})


    Problem_Definition_Agent = Agent(
        role='Problem_Definition_Agent',
        goal="""clarify the machine learning problem the user wants to solve""",
        backstory="""You are an expert in understanding and defining machine learning problems.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Data_Assessment_Agent = Agent(
        role='Data_Assessment_Agent',
        goal="""evaluate the data provided by the user""",
        backstory="""You specialize in data evaluation and preprocessing.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Model_Recommendation_Agent = Agent(
        role='Model_Recommendation_Agent',
        goal="""suggest the most suitable machine learning models""",
        backstory="""As an expert in machine learning algorithms, you recommend models that best fit the user's problem and data.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    Starter_Code_Generator_Agent = Agent(
        role='Starter_Code_Generator_Agent',
        goal="""generate starter Python code""",
        backstory="""You are a code wizard, able to generate starter code templates that users can customize for their projects.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    task_define_problem = Task(
        description=f"""Clarify and define the machine learning problem: {ml_problem}""",
        agent=Problem_Definition_Agent,
        expected_output="A clear and concise definition of the machine learning problem."
    )

    if data_upload:
        task_assess_data = Task(
            description=f"""Evaluate the user's data for quality and suitability""",
            agent=Data_Assessment_Agent,
            expected_output="An assessment of the data's quality and suitability, with suggestions for preprocessing or augmentation if necessary."
        )
    else:
        task_assess_data = Task(
            description=f"""The user has not uploaded any specific data""",
            agent=Data_Assessment_Agent,
            expected_output="A hypothetical dataset that might be useful for the user's machine learning problem, along with any necessary preprocessing steps."
        )

    task_recommend_model = Task(
        description=f"""Suggest suitable machine learning models""",
        agent=Model_Recommendation_Agent,
        expected_output="A list of suitable machine learning models for the defined problem and assessed data, along with the rationale for each suggestion."
    )

    task_generate_code = Task(
        description=f"""Generate starter Python code""",
        agent=Starter_Code_Generator_Agent,
        expected_output="Python code snippets for package import, data handling, model definition, and training, tailored to the user's project, plus a brief summary of the problem and model recommendations."
    )

    crew = Crew(
        agents=[Problem_Definition_Agent, Data_Assessment_Agent, Model_Recommendation_Agent, Starter_Code_Generator_Agent],
        tasks=[task_define_problem, task_assess_data, task_recommend_model, task_generate_code],
        verbose=False
    )

    try:
        result = crew.kickoff()
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": f"Error during task execution: {e}"})

    return templates.TemplateResponse("result.html", {"request": request, "result": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
