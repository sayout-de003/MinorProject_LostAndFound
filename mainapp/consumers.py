import json
from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth.models import User
from django.core.exceptions import PermissionDenied
from .models import ChatMessage, Match

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = f"chat_{self.room_name}"
        self.user = self.scope["user"]

        # Extract user IDs from the room name (Format: chat_{finder_id}_{owner_id})
        parts = self.room_name.split("_")
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            await self.close()
            return

        self.finder_id, self.owner_id = int(parts[1]), int(parts[2])

        # Ensure only the finder, owner, or admin can connect
        if not (self.user.is_superuser or self.user.id in [self.finder_id, self.owner_id]):
            await self.close()
            return

        # Join the chat room
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        # Leave chat room
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data["message"]
        sender_id = self.user.id

        # Ensure sender is authorized
        if sender_id not in [self.finder_id, self.owner_id] and not self.user.is_superuser:
            return

        # Save message in the database
        chat_message = ChatMessage.objects.create(
            match_id=self.room_name,
            sender=self.user,
            message=message
        )

        # Send message to WebSocket group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "chat_message",
                "message": message,
                "sender_id": sender_id,
                "timestamp": chat_message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            },
        )

    async def chat_message(self, event):
        message = event["message"]
        sender_id = event["sender_id"]
        timestamp = event["timestamp"]

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            "message": message,
            "sender_id": sender_id,
            "timestamp": timestamp
        }))
