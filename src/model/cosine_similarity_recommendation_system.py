"""
The system takes into account:

    1. Historical Performance:
        * Analyzes past project performance for similar task types
        * Considers project opportunity percentages and workload history

    2. Current Availability:
        * Tracks current workload vs maximum capacity
        * Excludes inactive employees

    3. Scoring System:
        * Combines performance history (70% weight) with availability (30% weight)
        * Normalizes scores across different project types
        * Customizable weights based on your priorities

The system uses Cosine similarity for project type matching (check _calculate_project_similarities for more information on that).
In addition, it uses StandardScaler for normalizing performance metrics.

You may adjust the custom scoring formula (70/30) for different results. This is just a test.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json
from datetime import datetime

class EmployeeRecommender:
    def __init__(self):
        self.employees_df = None
        self.projects_df = None
        self.performance_matrix = None
        self.project_similarity_matrix = None
        self.scaler = StandardScaler()
        
    def load_data(self, json_data: Dict):
        """Load and preprocess the JSON data."""
        # Extract employee information
        employees = []
        for emp in json_data['data']:
            employee_info = {
                'id': emp['id'],
                'first_name': emp['first_name'],
                'last_name': emp['last_name'],
                'type': emp['type'],
                'country': emp['country'],
                'current_load': self._calculate_current_load(emp['load']),
                'max_capacity': emp['max_capacity'],
                'is_inactive': emp['is_inactive']
            }
            employees.append(employee_info)
        
        self.employees_df = pd.DataFrame(employees)
        
        # Extract project information and create performance metrics
        projects = []
        for emp in json_data['data']:
            for proj in emp['projects']:
                project_info = {
                    'employee_id': emp['id'],
                    'project_id': proj['id'],
                    'project_type': proj['type'],
                    'opportunity_percentage': proj['opportunity_percentage'],
                    'avg_workload': self._calculate_avg_workload(proj['workload']),
                    'performance_score': self._calculate_performance_score(proj)
                }
                projects.append(project_info)
        
        self.projects_df = pd.DataFrame(projects)
        self._create_performance_matrix()
        self._calculate_project_similarities()
    
    def _calculate_current_load(self, load_data: List[Dict]) -> float:
        """Calculate current load based on recent entries."""
        if not load_data:
            return 0.0
        recent_loads = sorted(load_data, key=lambda x: x['timestamp'], reverse=True)
        return recent_loads[0]['load'] if recent_loads else 0.0
    
    def _calculate_avg_workload(self, workload_data: List[Dict]) -> float:
        """Calculate average workload for a project."""
        if not workload_data:
            return 0.0
        loads = [entry['load'] for entry in workload_data]
        return np.mean(loads) if loads else 0.0
    
    def _calculate_performance_score(self, project: Dict) -> float:
        """Calculate performance score based on workload and opportunity percentage."""
        avg_workload = self._calculate_avg_workload(project['workload'])
        opp_percentage = project['opportunity_percentage']
        return (avg_workload * 0.7 + opp_percentage * 0.3)
    
    def _create_performance_matrix(self):
        """Create a matrix of employee performance across different project types."""
        performance_pivot = self.projects_df.pivot_table(
            values='performance_score',
            index='employee_id',
            columns='project_type',
            aggfunc='mean',
            fill_value=0
        )
        self.performance_matrix = self.scaler.fit_transform(performance_pivot)
        self.performance_matrix = pd.DataFrame(
            self.performance_matrix,
            index=performance_pivot.index,
            columns=performance_pivot.columns
        )
    
    def _calculate_project_similarities(self):
        """
        Calculate similarity matrix between different project types.
        Cosine similarity measures the similarity between two vectors 
        of an inner product space. It is measured by the cosine of the 
        angle between two vectors and determines whether two vectors 
        are pointing in roughly the same direction.
        """
        # Transpose performance matrix to get project type vectors
        project_vectors = self.performance_matrix.T
        
        # Calculate cosine similarity between project types
        self.project_similarity_matrix = pd.DataFrame(
            cosine_similarity(project_vectors),
            index=project_vectors.index,
            columns=project_vectors.index
        )
    
    def get_similar_project_types(self, task_type: str, threshold: float = 0.5) -> List[str]:
        """Get list of similar project types based on cosine similarity."""
        if task_type not in self.project_similarity_matrix.index:
            return []
        
        similarities = self.project_similarity_matrix[task_type]
        similar_types = similarities[similarities >= threshold].index.tolist()
        return similar_types
    
    def recommend_employees(self, task_type: str, top_n: int = 3) -> List[Dict]:
        """
        Recommend top N employees for a given task type based on:
        1. Historical performance on similar projects (using cosine similarity)
        2. Current workload capacity
        3. Availability (not inactive)
        """
        # Get similar project types
        similar_types = self.get_similar_project_types(task_type)
        if not similar_types:
            raise ValueError(f"No historical data for task type: {task_type}")
        
        # Calculate weighted performance scores across similar project types
        weighted_scores = pd.Series(0.0, index=self.performance_matrix.index)
        for proj_type in similar_types:
            similarity_score = self.project_similarity_matrix.loc[task_type, proj_type]
            weighted_scores += self.performance_matrix[proj_type] * similarity_score
        
        # Normalize weighted scores
        weighted_scores = (weighted_scores - weighted_scores.min()) / (weighted_scores.max() - weighted_scores.min())
        
        # Combine with current workload and availability
        recommendations = []
        for emp_id, performance in weighted_scores.items():
            employee = self.employees_df[self.employees_df['id'] == emp_id].iloc[0]
            
            if employee['is_inactive']:
                continue
                
            # Calculate available capacity
            available_capacity = employee['max_capacity'] - employee['current_load']
            
            # Calculate final score (combining performance and availability)
            final_score = performance * 0.7 + (available_capacity / employee['max_capacity']) * 0.3
            
            recommendations.append({
                'employee_id': emp_id,
                'name': f"{employee['first_name']} {employee['last_name']}",
                'performance_score': performance,
                'available_capacity': available_capacity,
                'final_score': final_score,
                'similar_projects': similar_types
            })
        
        # Sort by final score and return top N
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        return recommendations[:top_n]

    def update_employee_load(self, employee_id: int, new_load: float):
        """Update an employee's current load."""
        idx = self.employees_df.index[self.employees_df['id'] == employee_id][0]
        self.employees_df.at[idx, 'current_load'] = new_load

