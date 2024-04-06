from dotenv import load_dotenv
load_dotenv()

#import markdown as md 
from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from tasks import MeetingPreparationTasks
from agents import MeetingPreparationAgents

tasks = MeetingPreparationTasks()
agents = MeetingPreparationAgents()

print("## Welcome to the Meeting Prep Crew")
print('-------------------------------')
participants = input("What are the emails for the participants (other than you) in the meeting?\n")
context = input("What is the context of the meeting?\n")
objective = input("What is your objective for this meeting?\n")



# Create Agents
researcher_agent = agents.research_agent()
industry_analyst_agent = agents.industry_analysis_agent()
meeting_strategy_agent = agents.meeting_strategy_agent()
summary_and_briefing_agent = agents.summary_and_briefing_agent()

# Create Tasks
research = tasks.research_task(researcher_agent, participants, context)
industry_analysis = tasks.industry_analysis_task(industry_analyst_agent, participants, context)
meeting_strategy = tasks.meeting_strategy_task(meeting_strategy_agent, context, objective)
summary_and_briefing = tasks.summary_and_briefing_task(summary_and_briefing_agent, context, objective)

meeting_strategy.context = [research, industry_analysis]
summary_and_briefing.context = [research, industry_analysis, meeting_strategy]

# Create Crew responsible for Copy
crew = Crew(
	agents=[
		researcher_agent,
		industry_analyst_agent,
		meeting_strategy_agent,
		summary_and_briefing_agent
	],
	tasks=[
		research,
		industry_analysis,
		meeting_strategy,
		summary_and_briefing
	],
 verbose=2,
 process=Process.hierarchical,
 manager_llm = ChatOpenAI(model_name = 'gpt-4-turbo-preview', temperature = 0.4),
)

result = crew.kickoff()


# Print results
print("\n\n################################################")
print("## Here is the result")
print("################################################\n")
print(result)

################################################
## Here is the result
################################################

# The briefing document for the ECB meeting on defining an AI strategy is structured as follows:
# 
# **1. Introduction**
# - Brief overview of the meeting's purpose: To define preliminary steps and deliverables for integrating AI into ECB operations, with a focus on navigating technological advancements and regulatory challenges.
# 
# **2. Participant Bios**
# - Include short bios of key participants, emphasizing their expertise and roles in AI strategy, financial regulation, and technology innovation. (Note: Specific bios are not provided here due to the earlier mentioned technical limitations.)
# 
# **3. Industry Overview**
# - **Technological Advancements**: Outline the potential impact of AI technologies like ChatGPT on the financial services landscape, highlighting opportunities for operational enhancements in customer service and portfolio management.
# - **Regulatory Challenges**: Discuss the complexities of the EU’s AI Act, divergent regulatory approaches between the EU and the U.S., and the importance of learning from other jurisdictions like the UK to navigate the regulatory landscape effectively.
# 
# **4. Talking Points**
# - Importance of AI Integration: Stress the necessity of adopting AI to maintain competitiveness and efficiency.
# - Navigating Regulatory Challenges: Emphasize the need for a strategic approach to comply with the EU’s AI Act while leveraging AI advancements.
# - Learning from Global Practices: Highlight the opportunity to inform the ECB's strategy with insights from the regulatory frameworks and AI integration strategies of other jurisdictions.
# 
# **5. Strategic Questions**
# - Discuss obstacles to AI integration, strategies for leveraging AI in operations, ensuring regulatory compliance, lessons from other jurisdictions, and actionable steps for compliance and innovation.
# 
# **6. Discussion Angles**
# - Explore potential areas for AI integration, compliance strategies with the EU's AI Act, international learnings, applications of AI to enhance operations, and potential partnerships.
# 
# **7. Strategic Recommendations**
# - **Overcoming Integration Hurdles**: Recommend establishing a task force to address technological and regulatory barriers.
# - **Leveraging AI for Enhanced Operations**: Suggest pilot projects to explore AI applications in customer service and portfolio management.
# - **Regulatory Navigation**: Propose a compliance roadmap that aligns with the EU’s AI Act and explores flexibility within regulatory frameworks.
# - **Learning from Others**: Advise on setting up exchange programs with institutions in jurisdictions like the UK to gain insights into their AI strategies.
# - **Actionable Steps for Compliance and Innovation**: Outline a phased approach for AI integration, starting with low-risk areas and gradually expanding as regulatory and technological landscapes evolve.
# 
# This document aims to equip the meeting participants with a comprehensive understanding of the current AI landscape, the regulatory environment, and strategic considerations for the ECB's AI strategy.