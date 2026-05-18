from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import supabase
from auth import get_current_user

router = APIRouter(
    tags=["projects"]
)  

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectSettings(BaseModel):
    embedding_model: str
    rag_strategy: str
    agent_type: str
    chunks_per_search: int
    final_context_size: int
    similarity_threshold: float
    number_of_queries: int
    reranking_enabled: bool
    reranking_model: str
    vector_weight: float
    keyword_weight: float

@router.get("/api/projects") 
def get_projects(clerk_id: str = Depends(get_current_user)): 
    try:
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()
 
        return { 
            "message": "Projects retrieved successfully",
            "data": result.data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")


@router.post("/api/projects") 
def create_project(project: ProjectCreate, clerk_id: str = Depends(get_current_user)):
    try:

        # Insert new project into database
        project_result = supabase.table("projects").insert({
            "name": project.name, 
            "description": project.description,
            "clerk_id": clerk_id
        }).execute()

        if not project_result.data:
            raise HTTPException(status_code=500, detail="Failed to create project")

        created_project = project_result.data[0]
        project_id = created_project["id"]

        # Create default settings for the project 
        settings_result = supabase.table("project_settings").insert({
            "project_id": project_id, 
            "embedding_model": "text-embedding-3-large",
            "rag_strategy": "basic",
            "agent_type": "agentic",
            "chunks_per_search": 10 ,
            "final_context_size": 5,
            "similarity_threshold": 0.3,
            "number_of_queries": 5,
            "reranking_enabled": True,
            "reranking_model": "rerank-english-v3.0",
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }).execute()

        if not settings_result.data:
            # If settings creation files, we should clean up the project
            supabase.table("projects").delete().eq("id", project_id).execute()
            raise HTTPException(status_code=500, detail="Failed to create project settings")

        return {
            "message": "Project created successfully", 
            "data": created_project
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to create project: {str(e)}")


@router.delete("/api/projects/{project_id}")
def delete_project(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Verify project exists and belongs to user 
        project_result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data: 
            raise HTTPException(status_code=404, detail="Project not found or access denied")

        # Delete project (CASCADE handles all related data)
        deleted_result = supabase.table("projects").delete().eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not deleted_result.data: 
            raise HTTPException(status_code=500, detail="Failed to delete project")

        return {
            "message": "Project deleted successfully", 
            "data": deleted_result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to delete project: {str(e)}")


@router.get("/api/projects/{project_id}")
async def get_project(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "message": "Project retrieved successfully", 
            "data": result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to get project: {str(e)}")


@router.get("/api/projects/{project_id}/chats")
async def get_project_chats(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("chats").select("*").eq("project_id", project_id).eq("clerk_id", clerk_id).order("created_at", desc=True).execute()

        return {
            "message": "Project retrieved successfully", 
            "data": result.data or []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to get project: {str(e)}")


@router.get("/api/projects/{project_id}/settings")
async def get_project_settings(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):

    try: 
       
        settings_result = supabase.table("project_settings").select("*").eq("project_id", project_id).execute()

        if not settings_result.data:
            raise HTTPException(status_code=404, detail="Project settings not found")

        return {
            "message": "Project settings retrieved successfully", 
            "data": settings_result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to get project settings: {str(e)}")


@router.put("/api/projects/{project_id}/settings")
async def update_project_settings(
    project_id: str, 
    settings: ProjectSettings, 
    clerk_id: str = Depends(get_current_user)
): 
    try: 
        # First verify the project exists and belongs to the user
        project_result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()    

        if not project_result.data:
            raise HTTPException(status_code=404, detail = f"Project not found or access denied")

        # Perform the update
        result = supabase.table("project_settings").update(settings.model_dump()).eq("project_id", project_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail = f"Project settings not found")

        return {
            "message": "Project settings updated successfully", 
            "data": result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to update project settings: {str(e)}")
