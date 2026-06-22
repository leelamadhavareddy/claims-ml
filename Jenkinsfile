// Sprint 6 CI/CD: test -> build image -> push to ECR -> deploy to ECS.
pipeline {
    agent any

    environment {
        AWS_REGION   = 'ap-south-1'
        ECR_REGISTRY = '525409063064.dkr.ecr.ap-south-1.amazonaws.com'
        IMAGE_NAME   = 'claims-ml'
        ECS_CLUSTER  = 'claims-clusters'
        ECS_SERVICE  = 'claims-service'
    }

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }

        stage('Test') {
            steps {
                sh 'pip3 install --break-system-packages -e ".[dev]"'
                sh 'python3 -m pytest -q'
            }
        }

        stage('Build image') {
            steps {
                sh 'docker build -t $IMAGE_NAME:$BUILD_NUMBER .'
            }
        }

        stage('Push to ECR') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'aws-creds',
                    usernameVariable: 'AWS_ACCESS_KEY_ID',
                    passwordVariable: 'AWS_SECRET_ACCESS_KEY')]) {
                    sh '''
                        aws ecr get-login-password --region $AWS_REGION \
                          | docker login --username AWS --password-stdin $ECR_REGISTRY
                        docker tag $IMAGE_NAME:$BUILD_NUMBER $ECR_REGISTRY/$IMAGE_NAME:$BUILD_NUMBER
                        docker tag $IMAGE_NAME:$BUILD_NUMBER $ECR_REGISTRY/$IMAGE_NAME:latest
                        docker push $ECR_REGISTRY/$IMAGE_NAME:$BUILD_NUMBER
                        docker push $ECR_REGISTRY/$IMAGE_NAME:latest
                    '''
                }
            }
        }

        stage('Approve deploy') {
            steps {
                input message: 'Deploy this build to production?'
            }
        }

        stage('Deploy to ECS') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'aws-creds',
                    usernameVariable: 'AWS_ACCESS_KEY_ID',
                    passwordVariable: 'AWS_SECRET_ACCESS_KEY')]) {
                    sh '''
                        aws ecs update-service \
                          --cluster $ECS_CLUSTER \
                          --service $ECS_SERVICE \
                          --force-new-deployment \
                          --region $AWS_REGION
                    '''
                }
            }
        }
    }

    post {
        success { echo 'Deployed successfully.' }
        failure { echo 'Pipeline failed - not deployed.' }
    }
}